import json
import os
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch import _dynamo, Tensor
from get_model.dataset.zarr_dataset import ATACBERTDataset
from get_model.dataset.zarr_dataset import worker_init_fn_get
_dynamo.config.suppress_errors = True


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, warmup_steps, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.warmup_steps = warmup_steps
        self.end_decay_steps = max_decay_steps + warmup_steps
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        # if step is less than warmup_steps, then learning rate is increasing
        if self.last_step < self.warmup_steps:
            return [base_lr * (self.last_step / self.warmup_steps) for base_lr in self.base_lrs]
        # if step is greater than end_decay_steps, then learning rate is end_learning_rate
        if self.last_step > self.end_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]
        # if step is between warmup_steps and end_decay_steps, then learning rate is decreasing
        return [(base_lr - self.end_learning_rate) *
                ((1 - (self.last_step - self.warmup_steps) / self.max_decay_steps) ** (self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.end_decay_steps:
            decay_lrs = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


class ATACBERTTrainer(object):
    def __init__(self, data: ATACBERTDataset, model, rank, world_size, # DDP args
                 batch_size=64, pin_memory=True, num_workers=8, # data loader args
                 warmup_steps=10000, max_decay_steps=125000, update_freq=1, learning_rate=1e-4, end_learning_rate=1e-6, # optimizer args
                 log_dir="logs/", # log dir
                 ):
        # create a random sampler
        sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        # initialize the data loader
        # TODO: add validation data loader
        self.val_dataset = None
        self.val_dataloader = None
        self.train_dataset = data
        self.train_dataloader = DataLoader(
            data,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=True,
            sampler=sampler,
            worker_init_fn=worker_init_fn_get,
        )
        # initialize optimizer
        # according to RoBERTa, learning rate and bath size is correlated, with batch size 256, learning rate 1e-4
        # https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md
        self.model = model
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                               # adapted from RoBERTa: https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/config/pretraining/base.yaml
                               betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01, 
                               lr=learning_rate)
        # initialize scheduler with linear warmup and linear decay
        # in RoBERTa it has learning rate of 1e-4 and end learning rate of 0.
        # I think it might be better to use end learning rate of 1e-6, which is 0.01 * 1e-4
        self.scheduler = PolynomialLRDecay(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            max_decay_steps=max_decay_steps,
            end_learning_rate=end_learning_rate,
        )
        self.log_dir = log_dir
        self.optimizer_rank = rank
        self.device = f'cuda:{rank}'
        self.device_id = rank
        self.current_epoch = 0
        self.global_step = 0
        self.updated = False
        self.update_freq = update_freq
        self.writer = SummaryWriter(log_dir=f'{self.log_dir}/log/')
    
    def training_epoch_begin(self):
        self.train_iterator = iter(self.train_dataloader)
        # set model to train mode
        self.model.train()

    def training_epoch_end(self):
        self.train_iterator = None
        # TODO: reset loss dict
        # self._reset_losses_dict()
        self.current_epoch += 1
        # if self is not updated, means the last batch is not updated, then update it
        if not self.updated:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.updated = True
        # reset self.updated to False
        self.updated = False
    
    def write_model(self, epoch=None, step=None, save_optimizer=False, optimizer_rank=None):
        if save_optimizer:
            assert optimizer_rank is not None
        if epoch is None:
            if step is None:
                model_save_file_name = f"{self.log_dir}/model.epoch.{self.current_epoch}.step.{self.global_step}.pt"
                if save_optimizer:
                    optimizer_save_file_name = f"{self.log_dir}/optimizer.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt"
                    scheduler_save_file_name = f"{self.log_dir}/scheduler.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt"
            else:
                model_save_file_name = f"{self.log_dir}/model.step.{step}.pt"
                if save_optimizer:
                    optimizer_save_file_name = f"{self.log_dir}/optimizer.step.{step}.rank.{optimizer_rank}.pt"
                    scheduler_save_file_name = f"{self.log_dir}/scheduler.step.{step}.rank.{optimizer_rank}.pt"
        else:
            if step is None:
                model_save_file_name = f"{self.log_dir}/model.epoch.{epoch}.pt"
                if save_optimizer:
                    optimizer_save_file_name = f"{self.log_dir}/optimizer.epoch.{epoch}.rank.{optimizer_rank}.pt"
                    scheduler_save_file_name = f"{self.log_dir}/scheduler.epoch.{epoch}.rank.{optimizer_rank}.pt"
            else:
                model_save_file_name = f"{self.log_dir}/model.epoch.{epoch}.step.{step}.pt"
                if save_optimizer:
                    optimizer_save_file_name = f"{self.log_dir}/optimizer.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt"
                    scheduler_save_file_name = f"{self.log_dir}/scheduler.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt"
        if isinstance(self.model, DDP):
            if self.use_lora:
                state_dic = lora.lora_state_dict(self.model.module)
                # add output_model to state_dic
                output_model_state_dic = self.model.module.output_model.state_dict()
                for key, value in output_model_state_dic.items():
                    state_dic[f"module.output_model.{key}"] = value
                torch.save(state_dic, model_save_file_name)
            else:
                torch.save(self.model.module.state_dict(), model_save_file_name)
        else:
            if self.use_lora:
                state_dic = lora.lora_state_dict(self.model)
                # add output_model to state_dic
                output_model_state_dic = self.model.output_model.output_network.state_dict()
                for key, value in output_model_state_dic.items():
                    state_dic[f"output_model.output_network.{key}"] = value
                torch.save(state_dic, model_save_file_name)
            else:
                torch.save(self.model.state_dict(), model_save_file_name)
        if save_optimizer:
            torch.save(self.optimizer.state_dict(), optimizer_save_file_name)
            torch.save(self.scheduler.state_dict(), scheduler_save_file_name)
    
    def write_optimizer(self, epoch=None, step=None, optimizer_rank=None):
        if epoch is None:
            if step is None:
                optimizer_save_file_name = f"{self.log_dir}/optimizer.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt"
                scheduler_save_file_name = f"{self.log_dir}/scheduler.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt"
            else:
                optimizer_save_file_name = f"{self.log_dir}/optimizer.step.{step}.rank.{optimizer_rank}.pt"
                scheduler_save_file_name = f"{self.log_dir}/scheduler.step.{step}.rank.{optimizer_rank}.pt"
        else:
            if step is None:
                optimizer_save_file_name = f"{self.log_dir}/optimizer.epoch.{epoch}.rank.{optimizer_rank}.pt"
                scheduler_save_file_name = f"{self.log_dir}/scheduler.epoch.{epoch}.rank.{optimizer_rank}.pt"
            else:
                optimizer_save_file_name = f"{self.log_dir}/optimizer.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt"
                scheduler_save_file_name = f"{self.log_dir}/scheduler.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt"
        torch.save(self.optimizer.state_dict(), optimizer_save_file_name)
        torch.save(self.scheduler.state_dict(), scheduler_save_file_name)

    def load_model(self, epoch=None, step=None, update_count=False):
        # if epoch or step is 0, don't load model
        if (epoch is not None and epoch == 0) or (step is not None and step == 0):
            return
        if epoch is None:
            if step is None:
                _state_dict = torch.load(
                    f"{self.log_dir}/model.epoch.{self.current_epoch}.step.{self.global_step}.pt",
                    maplocation=self.device
                )
            else:
                _state_dict = torch.load(
                    f"{self.log_dir}/model.step.{step}.pt",
                    map_location=self.device
                )
                if update_count:
                    self.global_step = step
                    self.current_epoch = step // self.batchs_per_epoch
        else:
            if step is None:
                _state_dict = torch.load(
                    f"{self.log_dir}/model.epoch.{epoch}.pt",
                    map_location=self.device
                )
                if update_count:
                    self.current_epoch = epoch
                    self.global_step = epoch * self.batchs_per_epoch
            else:
                _state_dict = torch.load(
                    f"{self.log_dir}/model.epoch.{epoch}.step.{step}.pt",
                    map_location=self.device
                )
                if update_count:
                    self.current_epoch = epoch
                    self.global_step = step
        _state_dict_is_ddp = list(_state_dict.keys())[0].startswith("module.")
        if isinstance(self.model, DDP):
            if _state_dict_is_ddp:
                self.model.load_state_dict(_state_dict, strict=self.use_lora==False)
            else:
                self.model.module.load_state_dict(_state_dict, strict=self.use_lora==False)
        else:
            if _state_dict_is_ddp:
                # create new OrderedDict that does not contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in _state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.model.load_state_dict(new_state_dict, strict=self.use_lora==False)
            else:
                self.model.load_state_dict(_state_dict, strict=self.use_lora==False)

    def load_optimizer(self, epoch=None, step=None, optimizer_rank=0):
        if epoch is None:
            if step is None:
                optimizer_state_dict = torch.load(
                    f"{self.log_dir}/optimizer.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt",
                    maplocation=self.device
                )
                scheduler_state_dict = torch.load(
                    f"{self.log_dir}/scheduler.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt",
                    maplocation=self.device
                )
            else:
                optimizer_state_dict = torch.load(
                    f"{self.log_dir}/optimizer.step.{step}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
                scheduler_state_dict = torch.load(
                    f"{self.log_dir}/scheduler.step.{step}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
        else:
            if step is None:
                optimizer_state_dict = torch.load(
                    f"{self.log_dir}/optimizer.epoch.{epoch}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
                scheduler_state_dict = torch.load(
                    f"{self.log_dir}/scheduler.epoch.{epoch}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
            else:
                optimizer_state_dict = torch.load(
                    f"{self.log_dir}/optimizer.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
                scheduler_state_dict = torch.load(
                    f"{self.log_dir}/scheduler.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.scheduler.load_state_dict(scheduler_state_dict)
    
    def write_loss_log(self, stage, loss):
        if self.device_id is None:
            scalar_name = f"loss/{stage}"
        else:
            scalar_name = f"loss/ddp_rank.{self.device_id}.{stage}"
        self.writer.add_scalar(scalar_name, loss, self.global_step)
        if stage == "train" and self.device_id == 0:
            for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.writer.add_histogram('weights/'+tag, value.data.cpu().numpy(), self.global_step)
                    try:
                        # only add gradients if they are not None
                        if value.grad is not None:
                            self.writer.add_histogram('grads/'+tag, value.grad.data.cpu().numpy(), self.global_step)
                    except:
                        print(f"failed to add grad histogram for '{tag}' in counter: {self.global_step}")

    def train_step(self):
        batch = next(self.train_iterator)
        peak_tokens, peak_density, seq, mask_idx = batch
        out = self.model.forward(peak_tokens.to(self.device), peak_density.to(self.device))
        # loss is cross entropy
        # only calculate loss for the masked tokens, ignore the cls and eos tokens
        logits: Tensor = out['logits'][:, 1:-1]
        # only calculate loss for the masked tokens
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[2])[mask_idx.reshape(-1).to(self.device)],
            seq.reshape(-1)[mask_idx.reshape(-1)].to(self.device)
        )
        loss.backward()
        # write loss to tensorboard
        self.write_loss_log("train", loss)
        self.global_step += 1
        self.updated = False
        # if global step meets the update frequency, then update the optimizer and scheduler
        if self.global_step % self.update_freq == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.updated = True
        return logits.detach().cpu().numpy(), loss.detach().cpu().numpy()
    
    def validation_epoch_begin(self):
        self.val_iterator = iter(self.val_dataloader)
        # set model to eval mode
        self.model.eval()
    
    def validation_epoch_end(self, reset_train_loss=False):
        self.val_iterator = None
    
    def validation_step(self):
        pass


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '15423'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def data_distributed_parallel_gpu(rank, model, dataset, world_size, epochs, num_save_batches, num_save_epochs):
    # set up training processes
    # Currently have bug if batch size does not match
    global result_dict
    torch.set_num_threads(6)
    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)
    # set up the distributed training
    save_every_step = num_save_batches
    save_every_epoch = num_save_epochs
    setup(rank, world_size)
    device = f'cuda:{rank}'
    torch.cuda.set_per_process_memory_fraction(1.0, rank)
    # utilize torch compile
    model = torch.compile(model.to(device))
    print(f'Compiled model in rank {rank}')
    
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    ddp_model.train()
    
    trainer = ATACBERTTrainer(model=ddp_model, data=dataset, rank=rank, world_size=world_size,)
    print(f"number of trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}, " +
          f"percentage = {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) / sum(p.numel() for p in trainer.model.parameters())}")
    # begin training
    dist.barrier()
    with Join([trainer.model]):
        for i in range(epochs):
            epoch_start_time = time.time()
            train_finished = False
            trainer.training_epoch_begin()
            while not train_finished:
                try:
                    batch_start_time = time.time()
                    logits, loss = trainer.train_step()
                    batch_end_time = time.time()
                    print(f"Rank {rank} batch {trainer.global_step} time: {batch_end_time - batch_start_time}")
                    dist.barrier()
                    if trainer.global_step % save_every_step == 0:
                        if rank == 0:
                            trainer.write_model(step=trainer.global_step)
                        # validate every save_every_step steps
                        if trainer.val_dataset is not None:
                            val_finished = False
                            val_begin_time = time.time()
                            trainer.validation_epoch_begin()
                            while not val_finished:
                                try:
                                    trainer.validation_step()
                                except StopIteration:
                                    val_finished = True
                            val_end_time = time.time()
                            result_dict = trainer.validation_epoch_end(reset_train_loss=True)
                            print(f"Rank {rank} batch {trainer.global_step} result: {result_dict}")
                            with open(
                                    f"{trainer.log_dir}/result_dict.batch.{trainer.global_step}.ddp_rank.{rank}.json", "w"
                            ) as f:
                                json.dump(result_dict, f)
                            dist.barrier()
                            all_val_loss = []
                            for k in range(world_size):
                                with open(
                                        f"{trainer.log_dir}/result_dict.batch.{trainer.global_step}.ddp_rank.{k}.json", "r"
                                ) as f:
                                    if trainer.val_dataset is not None:
                                        all_val_loss.append(json.load(f)["val_loss"])
                                    else:
                                        # train is val
                                        all_val_loss.append(json.load(f)["train_loss"])
                            print(f"Batch {trainer.global_step} all val loss: {np.mean(all_val_loss)}")
                            print(f"Batch {trainer.global_step} val time: {val_end_time - val_begin_time}")
                        dist.barrier()
                except StopIteration:
                    train_finished = True
            dist.barrier()
            # validate every epoch
            if trainer.val_dataset is not None:
                val_finished = False
                trainer.validation_epoch_begin()
                while not val_finished:
                    try:
                        trainer.validation_step()
                        dist.barrier()
                    except StopIteration:
                        val_finished = True
                result_dict = trainer.validation_epoch_end()
                print(f"Rank {rank} epoch {i} result: {result_dict}")
                with open(f"{trainer.log_dir}/result_dict.epoch.{i}.ddp_rank.{rank}.json", "w") as f:
                    json.dump(result_dict, f)
                # take all val loss together
                dist.barrier()
            trainer.training_epoch_end()
            epoch_end_time = time.time()
            print(f"Epoch {i} time: ", epoch_end_time - epoch_start_time)
            dist.barrier()
            if trainer.current_epoch % save_every_epoch == 0:
                if rank == 0:
                    trainer.write_model(epoch=trainer.current_epoch, save_optimizer=True, optimizer_rank=rank)
                else:
                    trainer.write_optimizer(epoch=trainer.current_epoch, optimizer_rank=rank)
    cleanup()
    # return all_losses
    return trainer


