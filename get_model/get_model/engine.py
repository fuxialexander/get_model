# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import logging
import math
import sys
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import get_model.utils as utils
from get_model.utils import frange_cycle_linear
from get_model.metrics import score_pearsonr, score_r2, score_spearmanr
from timm.data import Mixup
from timm.utils import ModelEma
from get_model.dataset.zarr_dataset import get_mask_pos, get_padding_pos
from get_model.model.vae_networks import PositionalNorm, ResBlock, VAEEncoder, VAEDecoder


import torch
import torch.nn.functional as F
from torch.distributions import Multinomial

def pretrain_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    normalize_target: bool = True,
    log_writer=None,
    lr_scheduler=None,
    start_steps=0,
    lr_schedule_values=None,
    wd_schedule_values=None,
):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    loss_masked = nn.MSELoss()
    #loss_atac = nn.PoissonNLLLoss(log_input=False, reduction="mean")

    for step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, _, _, _ = batch
        if min(chunk_size)<0:
            continue
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]


        sample_track = sample_track.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        # mask has 0, 1, -10000
        mask_for_loss = get_mask_pos(mask)
        padding_mask = get_padding_pos(mask)
        mask_for_loss = mask_for_loss.to(device, non_blocking=True).bool()
        padding_mask = padding_mask.to(device, non_blocking=True).bool()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        # chunk_size = chunk_size.to(device, non_blocking=True)
        n_peaks = n_peaks.to(device, non_blocking=True)


        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output_masked, _, target = model(peak_seq, sample_track, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std)

            # target generation
            with torch.no_grad():
                unnorm_targets = target
                if normalize_target:
                    print('normalize target')
                    regions_squeeze = unnorm_targets
                    regions_norm = (
                        regions_squeeze - regions_squeeze.mean(dim=-2, keepdim=True)
                    ) / (
                        regions_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt()
                        + 1e-6
                    )
                    # we find that the mean is about 0.48 and standard deviation is about 0.08.
                    regions_embed = regions_norm
                else:
                    regions_embed = unnorm_targets

                B, _, C = regions_embed.shape

            mask_for_loss = mask_for_loss.unsqueeze(-1)
            loss_masked_value = loss_masked(input=output_masked*mask_for_loss, target=regions_embed*mask_for_loss)
            #loss_atac_value = loss_atac(atac, labels_atac)
            # print(loss_masked_value, loss_atac_value) # masked loss is around 5 times larger than atac loss
            loss = loss_masked_value #+ loss_atac_value * 5

            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None and isinstance(log_writer, utils.TensorboardLogger):
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
        elif log_writer is not None and isinstance(log_writer, utils.WandBLogger):
            log_writer.update(data={'loss': loss_value,
                 'loss_scale': loss_scale_value,
                 'lr': max_lr,
                 'min_lr': min_lr,
                 'epoch': epoch,
                 'weight_decay': weight_decay_value,
                 'grad_norm': grad_norm},
                step=it, print_freq=print_freq)

        torch.distributed.barrier()
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # torch.distributed.barrier()
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, other_labels, criterion, epoch, hic_matrix=None, args=None):
    if model._get_name()=='GETFinetuneChrombpNet' or args.model=='get_finetune_motif_chrombpnet':
        return train_chrombpnet(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, other_labels, criterion, hic_matrix=hic_matrix)
    elif model._get_name()=='GETFinetuneAuto' or args.model=='get_finetune_motif_autoencoder':
        """Adjust train_motif_only to decide whether train for seq_embedding or motif_embedding"""
        return train_motif_autoencoder(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, other_labels, criterion, hic_matrix=hic_matrix)
    elif model._get_name()=='Motif_VAE' or args.model=='get_finetune_motif_vae':
        if epoch <100:
            #kl_regularizer = frange_cycle_linear(300)[epoch]
            kl_regularizer = 0.0001
        else:
            kl_regularizer = 1.0
        #print(f"Beta rate is: {kl_regularizer}")
        return train_motif_vae(model, kl_regularizer, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, other_labels, criterion, hic_matrix=hic_matrix)
    else:
        return train_class_batch_exp(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, other_labels, criterion, hic_matrix=hic_matrix)


def train_class_batch_exp(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, other_labels, criterion, hic_matrix=None):
    device = peak_seq.device
    padding_mask = get_padding_pos(mask)
    mask_for_loss = 1-padding_mask
    padding_mask = padding_mask.to(device, non_blocking=True).bool()
    mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        atac, exp, confidence = model(peak_seq, sample_track, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels, hic_matrix=hic_matrix)

    B, R, N = exp.shape
    exp = exp * mask_for_loss
    exp_target = exp_target * mask_for_loss
    if confidence is not None:
        cosine_similarity = torch.cosine_similarity(exp.reshape(B, -1), exp_target.reshape(B, -1), dim=-1).detach()
        # use cosine similarity as a target for confidence header, map each element to a bin in (-1,1), 50 bin in total
        # bin 0 is -1, bin 49 is 1
        # bin 0-24 is negative, bin 25 is 0, bin 26-49 is positive
        # get the bin index
        confidence_target = ((cosine_similarity+1)/2*49).long()
        # clip the value to 0-49
        confidence_target = torch.clamp(confidence_target, 0, 49)
        # confidence is (B, R, 1)
        confidence_pred = confidence.mean(1).softmax(dim=-1)
        # cross entropy loss for confidence header
        loss_confidence = nn.CrossEntropyLoss()(confidence_pred, confidence_target)
    else:
        confidence_pred = None
        confidence_target = None
        loss_confidence = None
    indices = torch.where(mask_for_loss==1)
    exp = exp[indices[0], indices[1], :].flatten()
    exp_target = exp_target[indices[0], indices[1], :].flatten()
    loss_exp = criterion(exp, exp_target)

    if atac is not None:
        atac = atac * mask_for_loss
        indices = torch.where(mask_for_loss==1)
        atac = atac[indices[0], indices[1], :].flatten()
        atac_target = atac_target.unsqueeze(-1) * mask_for_loss
        atac_target = atac_target[indices[0], indices[1], :].flatten()
        loss_atac = criterion(atac, atac_target)
        loss = loss_exp + loss_atac 
    else:
        loss = loss_exp
    if confidence is not None:
        loss = loss + loss_confidence * 0.1
    # return loss, exp, exp_target, atac, atac_target, confidence_pred, confidence_target
    return {'loss': loss, 'loss_atac': loss_atac, 'loss_exp': loss_exp, 'loss_confidence': loss_confidence,
            'exp_pred': exp, 'exp_target': exp_target, 
            'atac_pred': atac, 'atac_target': atac_target, 
            'confidence_pred': confidence_pred, 'confidence_target': confidence_target}


def multinomial_nll(true_counts, logits):
    """
    Compute the multinomial negative log-likelihood in PyTorch
    Args:
      true_counts (Tensor): observed count values
      logits (Tensor): predicted logit values
    """
    counts_per_example = torch.sum(true_counts, dim=-1)
    # Creating a Multinomial distribution in PyTorch
    dist = Multinomial(total_count=counts_per_example, logits=logits)
    # Calculating the log probability and then the negative log-likelihood
    log_prob = dist.log_prob(true_counts)
    nll = -torch.sum(log_prob) / true_counts.size(0)
    return nll

def train_chrombpnet(model, peak_seq, aprofile_target, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atpm_target, exp_target, other_labels, criterion, hic_matrix=None):
    device = peak_seq.device
    padding_mask = get_padding_pos(mask)
    mask_for_loss = 1-padding_mask
    padding_mask = padding_mask.to(device, non_blocking=True).bool()
    mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        atpm, aprofile = model(peak_seq, aprofile_target, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels, hic_matrix=hic_matrix)

    B, R, N = atpm.shape
    atpm = atpm * mask_for_loss
    indices = torch.where(mask_for_loss==1)
    atpm = atpm[indices[0], indices[1], :].flatten()
    atpm_target = atpm_target.unsqueeze(-1) * mask_for_loss
    atpm_target = atpm_target[indices[0], indices[1], :].flatten()
    aprofile_target = aprofile_target
    loss_atpm = criterion(atpm.float(), atpm_target.float())
    loss_aprofile = criterion(aprofile.float(), aprofile_target.float())
    
    loss = loss_atpm + loss_aprofile 
    # return loss, atpm, atpm_target, aprofile, aprofile_target
    return {'loss': loss, 'loss_atpm': loss_atpm, 'loss_aprofile': loss_aprofile,
            'atpm_pred': atpm, 'atpm_target': atpm_target, 
            'aprofile_pred': aprofile, 'aprofile_target': aprofile_target}

def train_motif_autoencoder(model, peak_seq, aprofile_target, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atpm_target, exp_target, other_labels, criterion, hic_matrix=None):
    """ if 'train_motif_only' parameter is True, the model only trains for motif_embedding, using only motif_emb_loss
    the default is False, which trains for seq_embedding by calculating loss using sum of motif_emb_loss and seq_emb_loss  """
    device = peak_seq.device
    padding_mask = get_padding_pos(mask)
    mask_for_loss = 1-padding_mask
    padding_mask = padding_mask.to(device, non_blocking=True).bool()
    mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
    train_motif_only = True
    if train_motif_only == True:
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            target_motif_emb, pred_motif_emb, latent_emb, mu, logvar = model(peak_seq, aprofile_target, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels, hic_matrix=hic_matrix,  train_motif_only=train_motif_only)
        # Using MSE because it's a regression task
        loss_motif_emb = F.mse_loss(pred_motif_emb, target_motif_emb, reduction='mean')
        # Using KL divergence compare latent and normal distribution to prevent lazy learning
        KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_latent_emb = KLD_loss / mu.size(0)  # Average over the batch

        loss = loss_motif_emb + loss_latent_emb
        # return loss, atpm, atpm_target, aprofile, aprofile_target
        #print(f"loss: {loss}")
        return {'loss': loss, 'loss_motif_emb': loss_motif_emb, 'loss_latent_emb': loss_latent_emb,
                'pred_motif_emb': pred_motif_emb, 'target_motif_emb': target_motif_emb,
                'latent_emb':latent_emb, 'mu': mu, 'logvar': logvar}
    else:
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            # target_seq_emb, pred_seq_emb: [B, L, 4]; target_motif_emb, pred_motif_emb: [B, L, 639]; latent_emb: latent space learned using encoder
            target_seq_emb, pred_seq_emb, target_motif_emb, pred_motif_emb, latent_emb, mu, logvar = model(peak_seq, aprofile_target, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels, hic_matrix=hic_matrix,  train_motif_only=train_motif_only)
        #print("pred_seq_emb shape:", pred_seq_emb.shape)
        #print("target_seq_emb shape:", target_seq_emb.shape)
        #print("pred_motif_emb shape:", target_motif_emb.shape)
        #print("target_motif_emb shape:", target_motif_emb.shape)
        #print("latent_emb:",latent_emb.shape)
        
        # Using cross-entropy because it's a classification tasks
        loss_seq_emb = F.cross_entropy(pred_seq_emb.reshape(-1, 4), target_seq_emb.reshape(-1, 4))
        # Using MSE because it's a regression task
        loss_motif_emb = F.mse_loss(pred_motif_emb, target_motif_emb, reduction='mean')
        # Using KL divergence compare latent and normal distribution to prevent lazy learning
        KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_latent_emb = KLD_loss / mu.size(0)  # Average over the batch

        loss = loss_seq_emb + loss_motif_emb + loss_latent_emb
        # return loss, atpm, atpm_target, aprofile, aprofile_target
        #print(f"loss: {loss}")
        return {'loss': loss, 'loss_seq_emb': loss_seq_emb, 'loss_motif_emb': loss_motif_emb, 'loss_latent_emb': loss_latent_emb,
                'pred_seq_emb': pred_seq_emb, 'target_seq_emb': target_seq_emb, 
                'pred_motif_emb': pred_motif_emb, 'target_motif_emb': target_motif_emb,
                'latent_emb':latent_emb, 'mu': mu, 'logvar': logvar}

def train_motif_vae(model, kl_regularizer, peak_seq, aprofile_target, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atpm_target, exp_target, other_labels, criterion, hic_matrix=None):
    device = peak_seq.device
    padding_mask = get_padding_pos(mask)
    mask_for_loss = 1-padding_mask
    padding_mask = padding_mask.to(device, non_blocking=True).bool()
    mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        target_input, pred_input, latent_emb, latent_mu, latent_log_std, total_loss, recon_loss, kl_loss = model(peak_seq, aprofile_target, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, kl_regularizer, other_labels, hic_matrix=hic_matrix)
    
    # return loss, atpm, atpm_target, aprofile, aprofile_target
    #print(f"loss: {loss}")
    return {'loss': total_loss, 'reconstruction_loss': recon_loss, 'kl_loss': kl_loss,
            'pred_latent_emb': latent_emb, 'latent_mean': latent_mu,'latent_log_std':latent_log_std, 
            'target_input': target_input, 'pred_input': pred_input}


def finetune_train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    data_loader_val: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    args=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):  
        # logging.info("data_iter_step: {}".format(data_iter_step))
        # logging.info("start getting batch")
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels, hic_matrix = batch
        if min(chunk_size)<0:
            continue
        # logging.info("Got batch")
        step = data_iter_step // update_freq
        
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if (
            lr_schedule_values is not None
            or wd_schedule_values is not None
            and data_iter_step % update_freq == 0
        ):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # logging.info("start cuda computing")
        sample_track = sample_track.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        # chunk_size = chunk_size.to(device, non_blocking=True)
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).bfloat16()
        other_labels = other_labels.to(device, non_blocking=True).bfloat16()
        hic_matrix = hic_matrix.to(device, non_blocking=True).bfloat16()
        if loss_scaler is None:
            peaks = peaks.half()
        result = train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels[:,:,0], labels_data, other_labels, criterion, epoch, hic_matrix=hic_matrix, args=args)
        loss = result['loss']
        loss_value = loss.item()

        # logging.info("Got loss")
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]


        torch.cuda.synchronize()

        
        # NOTE: evaluation
        # EMA
        if (data_iter_step + 1) % update_freq == 0 and args.eval_each_step:
            test_stats = evaluate_all(
                data_loader_val, model, device, criterion, args, epoch = epoch, printlog=False
            )
        else:
            test_stats = None

        metric_logger.update(loss=loss_value)
        if test_stats is not None:
            metric_logger.update(r2score=test_stats["r2score"])
            metric_logger.update(pearsonr_score=test_stats["pearsonr_score"])
            metric_logger.update(spearmanr_score=test_stats["spearmanr_score"])
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None and isinstance(log_writer, utils.TensorboardLogger):
            log_writer.update(loss=loss_value, head="loss")
            if test_stats is not None:
                log_writer.update(r2score=test_stats["r2score"], head="loss")
                log_writer.update(
                    pearsonr_score=test_stats["pearsonr_score"], head="loss"
                )
                log_writer.update(
                    spearmanr_score=test_stats["spearmanr_score"], head="loss"
                )
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
        elif log_writer is not None and isinstance(log_writer, utils.WandBLogger):
            log_writer.update(data={
                'training_loss': loss_value,
                'loss_scale': loss_scale_value,
                'lr': max_lr,
                'min_lr': min_lr,
                'epoch': epoch,
                'weight_decay': weight_decay_value,
                'grad_norm': grad_norm},
                step=it, print_freq=print_freq)
            if test_stats is not None:
                log_writer.update(data={
                    'r2score': test_stats["r2score"],
                    'pearsonr_score': test_stats["pearsonr_score"],
                    'spearmanr_score': test_stats["spearmanr_score"],
                    'epoch': epoch},
                    step=it, print_freq=print_freq)
        
        # print("Finished one iteration")

    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def cal_score_stats(preds, obs, data_loader, args):

    if args.eval_nonzero:
        r2score = score_r2(preds[obs > 0], obs[obs > 0])
        spearmanr_score = score_spearmanr(preds[obs > 0], obs[obs > 0])
        pearsonr_score = score_pearsonr(preds[obs > 0], obs[obs > 0])
    elif args.eval_tss:
        r2score = score_r2(preds, obs)
        spearmanr_score = score_spearmanr(preds, obs)
        pearsonr_score = score_pearsonr(preds, obs)
    else:
        r2score = score_r2(preds, obs)
        spearmanr_score = score_spearmanr(preds, obs)
        pearsonr_score = score_pearsonr(preds, obs)

    return r2score, spearmanr_score, pearsonr_score


@torch.no_grad()
def evaluate_pretrain(data_loader, model, device, args, epoch=0, printlog=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    output_masked_list = []
    target_list = []
    for i, batch in enumerate(data_loader):
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, _, _, _ = batch
        if min(chunk_size)<0:
            continue
    # for i in tqdm(range(100)):
        mask_for_loss = get_mask_pos(mask)
        padding_mask = get_padding_pos(mask)
        mask_for_loss = mask_for_loss.to(device, non_blocking=True).bool()
        padding_mask = padding_mask.to(device, non_blocking=True).bool()
        peak_seq = peak_seq.bfloat16().cuda()
        sample_track = sample_track.bfloat16().cuda()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_masked, _, target = model.forward(peak_seq, sample_track, mask_for_loss, padding_mask, chunk_size, n_peaks.cuda(), max_n_peaks, motif_mean_std.cuda())
        normalize_target = False
        unnorm_targets = target
        if normalize_target:
            regions_squeeze = unnorm_targets
            regions_norm = (
                regions_squeeze - regions_squeeze.mean(dim=-2, keepdim=True)
            ) / (
                regions_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt()
                + 1e-6
            )
            # we find that the mean is about 0.48 and standard deviation is about 0.08.
            regions_embed = regions_norm
        else:
            regions_embed = unnorm_targets

        B, _, C = regions_embed.shape
        mask_for_loss = mask_for_loss.unsqueeze(-1)
        loss_masked = nn.MSELoss()
        loss_masked_value = loss_masked(input=output_masked*mask_for_loss, target=regions_embed*mask_for_loss)
        #loss_atac_value = loss_atac(atac, labels_atac)
        # print(loss_masked_value, loss_atac_value) # masked loss is around 5 times larger than atac loss
        loss = loss_masked_value #+ loss_atac_value * 5
        target = (regions_embed*mask_for_loss).float().detach().cpu().numpy().flatten()
        output_masked = (output_masked*mask_for_loss).float().detach().cpu().numpy().flatten()
        output_masked = output_masked[target>0]
        target = target[target>0]
        output_masked_list.append(output_masked)
        target_list.append(target)
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

    output_masked_list = np.concatenate(output_masked_list).flatten()
    target_list = np.concatenate(target_list).flatten()
    r2score, pearsonr_score, spearmanr_score = cal_score_stats(output_masked_list, target_list, data_loader, args)

    metric_logger.meters["r2score"].update(r2score, n=1)
    metric_logger.meters["pearsonr_score"].update(pearsonr_score, n=1)
    metric_logger.meters["spearmanr_score"].update(spearmanr_score, n=1)

    if printlog:
        print(
            "* Score@R2 {r2:.3f} Score@pearsonr {pearson:.3f} Score@spearmanr {spearman:.3f}  loss {losses.global_avg:.3f}".format(
                r2=r2score,
                pearson=pearsonr_score,
                spearman=spearmanr_score,
                losses=metric_logger.loss,
            )
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    
@torch.no_grad()
def evaluate_all(data_loader, model, device, criterion, args, epoch=0, printlog=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    preds = []
    obs = []
    preds_atac = []
    obs_atac = []
    preds_atpm = []
    obs_atpm = []
    preds_aprofile = []
    obs_aprofile = []

    #Below is for motif autoencoder
    preds_seq_emb = []
    obs_seq_emb = []
    preds_motif_emb = []
    obs_motif_emb = []
    preds_input_emb = []
    obs_input_emb = []

    from tqdm import tqdm
    for batch in tqdm(data_loader):
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels, hic_matrix = batch
        if min(chunk_size)<0:
            continue
        sample_track = sample_track.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        # chunk_size = chunk_size.to(device, non_blocking=True)
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).bfloat16()
        other_labels = other_labels.to(device, non_blocking=True).bfloat16()
        # compute output
        atac_targets = other_labels[:,:,0]

        #if model._get_name()=='Motif_VAE' or args.model=='get_finetune_motif_vae':
            #print(f"evaluation epoch is: {epoch}")

        result = train_class_batch(
            model=model, 
            peak_seq=peak_seq, 
            sample_track=sample_track,
            mask=mask,
            chunk_size=chunk_size,
            n_peaks=n_peaks,
            max_n_peaks=max_n_peaks,
            motif_mean_std=motif_mean_std,
            atac_target=atac_targets,
            exp_target=labels_data,
            other_labels=other_labels,
            criterion=criterion,
            epoch = epoch,
            hic_matrix=hic_matrix,
            args=args
        )

        if model._get_name() =='GETFinetuneChrombpNet' or args.model=='get_finetune_motif_chrombpnet':
            loss = result['loss'].item()
            loss_atpm = result['loss_atpm'].item()
            loss_aprofile = result['loss_aprofile'].item()
            atpm = result['atpm_pred']
            atpm_target = result['atpm_target']
            aprofile = result['aprofile_pred']
            aprofile_target = result['aprofile_target']
            preds_atpm.append(atpm.float().reshape(-1).detach().cpu().numpy())
            obs_atpm.append(atpm_target.float().reshape(-1).detach().cpu().numpy())
            preds_aprofile.append(aprofile.float().reshape(-1).detach().cpu().numpy())
            obs_aprofile.append(aprofile_target.float().reshape(-1).detach().cpu().numpy())

            metric_logger.update(loss=loss)
            metric_logger.update(loss_atpm=loss_atpm)
            metric_logger.update(loss_aprofile=loss_aprofile)
        elif model._get_name()=='Motif_VAE' or args.model=='get_finetune_motif_vae':
            loss = result['loss'].item()
            loss_recon = result['reconstruction_loss'].item()
            loss_kl = result['kl_loss'].item()

            target_input = result['target_input']
            pred_input = result['pred_input']

            preds_input_emb.append(pred_input.float().transpose(1,2).reshape(-1).detach().cpu().numpy()) # [B, L, 640] -> [B, 640, L]-> [B*640*L, ]
            obs_input_emb.append(target_input.float().transpose(1,2).reshape(-1).detach().cpu().numpy())

            metric_logger.update(loss=loss)
            metric_logger.update(loss_atpm=loss_recon)
            metric_logger.update(loss_aprofile=loss_kl)
        elif model._get_name()=='GETFinetuneAuto' or args.model=='get_finetune_motif_autoencoder':
            """Define train_motif_only. If train_motif_only is True, train for motif_embeddings; if False, train for sequence_embeddings"""
            train_motif_only = True
            if train_motif_only == False:
                # Extracting all the required items from the result dictionary
                loss = result['loss'].item()  # Total loss
                loss_seq_emb = result['loss_seq_emb'].item()  # Loss for sequence embeddings
                loss_motif_emb = result['loss_motif_emb'].item()  # Loss for motif embeddings
                loss_latent_emb = result['loss_latent_emb'].item()  # KL divergence loss for VAE
                
                # For this example, we're assuming your model might also return predicted and target sequence embeddings (or similar outputs)
                pred_seq_emb = result['pred_seq_emb']
                target_seq_emb = result['target_seq_emb']
                pred_motif_emb = result['pred_motif_emb']
                target_motif_emb = result['target_motif_emb']
                latent_emb = result['latent_emb']

                # Appending the predictions and targets for further analysis or metric computation
                preds_seq_emb.append(pred_seq_emb.float().reshape(-1).detach().cpu().numpy())
                obs_seq_emb.append(target_seq_emb.float().reshape(-1).detach().cpu().numpy())
                preds_motif_emb.append(pred_motif_emb.float().reshape(-1).detach().cpu().numpy())
                obs_motif_emb.append(target_motif_emb.float().reshape(-1).detach().cpu().numpy())
    
                # Logging the loss components for monitoring
                metric_logger.update(loss=loss)
                metric_logger.update(loss_seq_emb=loss_seq_emb)
                metric_logger.update(loss_motif_emb=loss_motif_emb)
                metric_logger.update(loss_latent_emb=loss_latent_emb)
            else:
                # Extracting all the required items from the result dictionary
                loss = result['loss'].item()  # Total loss
                loss_motif_emb = result['loss_motif_emb'].item()  # Loss for motif embeddings
                loss_latent_emb = result['loss_latent_emb'].item()  # KL divergence loss for VAE
                
                # For this example, we're assuming your model might also return predicted and target sequence embeddings (or similar outputs)
                pred_motif_emb = result['pred_motif_emb']
                target_motif_emb = result['target_motif_emb']
                latent_emb = result['latent_emb']

                # Appending the predictions and targets for further analysis or metric computation
                preds_motif_emb.append(pred_motif_emb.float().reshape(-1).detach().cpu().numpy())
                obs_motif_emb.append(target_motif_emb.float().reshape(-1).detach().cpu().numpy())
    
                # Logging the loss components for monitoring
                metric_logger.update(loss=loss)
                metric_logger.update(loss_motif_emb=loss_motif_emb)
                metric_logger.update(loss_latent_emb=loss_latent_emb)
        else:
            exp = result['exp_pred']
            exp_target = result['exp_target']
            atac = result['atac_pred']
            atac_target = result['atac_target']
            loss = result['loss'].item()
            loss_atac = result['loss_atac'].item()
            loss_exp = result['loss_exp'].item()
            loss_confidence = result['loss_confidence'].item()
            if args.eval_tss:
                padding_mask = get_padding_pos(mask)
                mask_for_loss = 1-padding_mask
                padding_mask = padding_mask.to(device, non_blocking=True).bool()
                mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
                indices = torch.where(mask_for_loss==1)
                # other_labels is B, R, N where [:,:, 1] is TSS indicator
                other_labels_reshape = other_labels[indices[0], indices[1], 1].flatten()
                preds.append(exp.reshape(-1, 2)[other_labels_reshape==1, :].reshape(-1).detach().cpu().numpy())
                obs.append(exp_target.reshape(-1,2)[other_labels_reshape==1, :].reshape(-1).detach().cpu().numpy())
            else:
                preds.append(exp.reshape(-1).detach().cpu().numpy())
                obs.append(exp_target.reshape(-1).detach().cpu().numpy())
            preds_atac.append(atac.reshape(-1).detach().cpu().numpy())
            obs_atac.append(atac_target.reshape(-1).detach().cpu().numpy())

            metric_logger.update(loss=loss)
            metric_logger.update(loss_atac=loss_atac)
            metric_logger.update(loss_exp=loss_exp)
            metric_logger.update(loss_confidence=loss_confidence)

    if model._get_name()=='GETFinetuneChrombpNet' or args.model=='get_finetune_motif_chrombpnet':
        preds_atpm = np.concatenate(preds_atpm, axis=0).reshape(-1)
        obs_atpm = np.concatenate(obs_atpm, axis=0).reshape(-1)
        preds_aprofile = np.concatenate(preds_aprofile, axis=0).reshape(-1)
        obs_aprofile = np.concatenate(obs_aprofile, axis=0).reshape(-1)
        bin=100
        obs_aprofile = np.array([np.mean(obs_aprofile[i:i+bin]) for i in range(0, len(obs_aprofile), bin)])
        preds_aprofile = np.array([np.mean(preds_aprofile[i:i+bin]) for i in range(0, len(preds_aprofile), bin)])
        r2score_atpm, pearsonr_score_atpm, spearmanr_score_atpm = cal_score_stats(preds_atpm, obs_atpm, data_loader, args)
        r2score_aprofile, pearsonr_score_aprofile, spearmanr_score_aprofile = cal_score_stats(preds_aprofile, obs_aprofile, data_loader, args)

        metric_logger.meters["r2score"].update(r2score_atpm, n=1)
        metric_logger.meters["pearsonr_score"].update(pearsonr_score_atpm, n=1)    
        metric_logger.meters["spearmanr_score"].update(spearmanr_score_atpm, n=1)
        metric_logger.meters["r2score_atac"].update(r2score_aprofile, n=1)
        metric_logger.meters["pearsonr_score_atac"].update(pearsonr_score_aprofile, n=1)
        metric_logger.meters["spearmanr_score_atac"].update(spearmanr_score_aprofile, n=1)


        if printlog:
            print(
                "* Score@R2 {r2:.3f} Score@pearsonr {pearson:.3f} Score@spearmanr {spearman:.3f} \n Score@R2 {r2score_aprofile:.3f} Score@pearsonr {pearsonr_score_aprofile:.3f} Score@spearmanr {spearmanr_score_aprofile:.3f}  \n total_loss {loss:.3f} reconstruction_loss {loss_recon:.3f}, kl_loss {loss_kl:.3f}".format(
                    r2=r2score_atpm,
                    pearson=pearsonr_score_atpm,
                    spearman=spearmanr_score_atpm,
                    r2score_aprofile=r2score_aprofile,
                    pearsonr_score_aprofile=pearsonr_score_aprofile,
                    spearmanr_score_aprofile=spearmanr_score_aprofile,
                    loss = loss,
                    loss_recon=loss_recon,
                    loss_kl=loss_kl,
                )
            )
    elif model._get_name()=='Motif_VAE' or args.model=='get_finetune_motif_vae':
        # Cal correlation scores for 639 motif embeddings
        print("1/3 Concatenate Embeddings")
        preds_input_emb = np.concatenate(preds_input_emb, axis=0).reshape(-1)
        obs_input_emb = np.concatenate(obs_input_emb, axis=0).reshape(-1)
        #***********************************************
        #bin=80
        #preds_input_emb = np.concatenate(preds_input_emb, axis=0).reshape(-1)
        #num_complete_bins = preds_input_emb.size // bin
        #print(f"Divide embeddings from {preds_input_emb.size} into {num_complete_bins} regions")
        #if num_complete_bins > 0:
        #    complete_bins = preds_input_emb[:num_complete_bins * bin].reshape(-1, bin)
        #    complete_bin_means = complete_bins.mean(axis=1)
        #if preds_input_emb.size % bin > 0:
        #    remainder = preds_input_emb[num_complete_bins * bin:]
        #    remainder_mean = np.array([remainder.mean()])  
        #    preds_input_aprofile = np.concatenate([complete_bin_means, remainder_mean])
        #else:
        #    preds_input_aprofile = complete_bin_means

        #obs_input_emb = np.concatenate(obs_input_emb, axis=0).reshape(-1)
        #num_complete_bins = obs_input_emb.size // bin
        #if num_complete_bins > 0:
        #    complete_bins = obs_input_emb[:num_complete_bins * bin].reshape(-1, bin)
        #    complete_bin_means = complete_bins.mean(axis=1)
        #if obs_input_emb.size % bin > 0:
        #    remainder = obs_input_emb[num_complete_bins * bin:]
        #    remainder_mean = np.array([remainder.mean()])  
        #    obs_input_aprofile = np.concatenate([complete_bin_means, remainder_mean])
        #else:
        #    obs_input_aprofile = complete_bin_means

        print("2/3 Calculating embedding correlation scores")
        print(f'Embedding shape: {preds_input_emb.shape}')
        #r2score_input, pearsonr_score_input, spearmanr_score_input = cal_score_stats(preds_input_aprofile, obs_input_aprofile, data_loader, args)
        r2score_input, pearsonr_score_input, spearmanr_score_input = cal_score_stats(preds_input_emb, obs_input_emb, data_loader, args)

        
        #***********************************************
        print('3/3 Updating metric scores')
        metric_logger.meters["r2score"].update(r2score_input, n=1)
        metric_logger.meters["pearsonr_score"].update(pearsonr_score_input, n=1)    
        metric_logger.meters["spearmanr_score"].update(spearmanr_score_input, n=1)
        metric_logger.meters["r2score_atac"].update(0.0, n=1)
        metric_logger.meters["pearsonr_score_atac"].update(0.0, n=1)
        metric_logger.meters["spearmanr_score_atac"].update(0.0, n=1)
        print('Done!')


        if printlog:
            print(
                "* Score@R2 {r2:.3f} Score@pearsonr {pearson:.3f} Score@spearmanr {spearman:.3f} \n Score@R2 {r2score_aprofile:.3f} Score@pearsonr {pearsonr_score_aprofile:.3f} Score@spearmanr {spearmanr_score_aprofile:.3f} \n total_loss {loss:.3f} recon_loss {loss_recon:.3f} kl_loss {loss_kl:.3f}".format(
                    r2=r2score_input,
                    pearson=pearsonr_score_input,
                    spearman=spearmanr_score_input,
                    r2score_aprofile=0.0,
                    pearsonr_score_aprofile=0.0,
                    spearmanr_score_aprofile=0.0,
                    loss=loss,
                    loss_recon = loss_recon,
                    loss_kl=loss_kl,
                )
            )
    elif model._get_name()=='GETFinetuneAuto' or args.model=='get_finetune_motif_autoencoder':
        train_motif_only = True
        """If train_motif_only is True, train for motif_embeddings only"""
        if train_motif_only == False:
            print("1/5 Concatenate embeddings")
            preds_seq_emb = np.concatenate(preds_seq_emb, axis=0).reshape(-1)
            obs_seq_emb = np.concatenate(obs_seq_emb, axis=0).reshape(-1)
            #preds_motif_emb = np.concatenate(preds_motif_emb, axis=0).reshape(-1)
            #obs_motif_emb = np.concatenate(obs_motif_emb, axis=0).reshape(-1)
            bin=100
            print("2/5 Compress motif_embeddings")
            #preds_motif_aprofile = np.array([np.mean(preds_motif_emb[i:i+bin]) for i in range(0, len(preds_motif_emb), bin)])
            #obs_motif_aprofile = np.array([np.mean(obs_motif_emb[i:i+bin]) for i in range(0, len(obs_motif_emb), bin)])

            #***********************************************
            preds_motif_emb = np.concatenate(preds_motif_emb, axis=0)
            num_complete_bins = preds_motif_emb.size // bin
            if num_complete_bins > 0:
                complete_bins = preds_motif_emb[:num_complete_bins * bin].reshape(-1, bin)
                complete_bin_means = complete_bins.mean(axis=1)
            if preds_motif_emb.size % bin > 0:
                remainder = preds_motif_emb[num_complete_bins * bin:]
                remainder_mean = np.array([remainder.mean()])  
                preds_motif_aprofile = np.concatenate([complete_bin_means, remainder_mean])
            else:
                preds_motif_aprofile = complete_bin_means

            obs_motif_emb = np.concatenate(obs_motif_emb, axis=0)
            num_complete_bins = obs_motif_emb.size // bin
            if num_complete_bins > 0:
                complete_bins = obs_motif_emb[:num_complete_bins * bin].reshape(-1, bin)
                complete_bin_means = complete_bins.mean(axis=1)
            if obs_motif_emb.size % bin > 0:
                remainder = obs_motif_emb[num_complete_bins * bin:]
                remainder_mean = np.array([remainder.mean()])  
                obs_motif_aprofile = np.concatenate([complete_bin_means, remainder_mean])
            else:
                obs_motif_aprofile = complete_bin_means
            #***********************************************
            print('3/5 Calculating sequence embedding metric scores')
            print(f'preds_seq_emb.shape: {preds_seq_emb.shape}')
            r2score_seq_emb, pearsonr_score_seq_emb, spearmanr_score_seq_emb = cal_score_stats(preds_seq_emb, obs_seq_emb, data_loader, args)
            print('4/5 Calculating motif embedding metric scores')
            print(f'preds_motif_emb.shape: {preds_motif_aprofile.shape}')
            r2score_motif_emb, pearsonr_score_motif_emb, spearmanr_score_motif_emb = cal_score_stats(preds_motif_aprofile, obs_motif_aprofile, data_loader, args)
            print('5/5 Updating metric scores')
            metric_logger.meters["r2score"].update(r2score_seq_emb, n=1)
            metric_logger.meters["pearsonr_score"].update(pearsonr_score_seq_emb, n=1)    
            metric_logger.meters["spearmanr_score"].update(spearmanr_score_seq_emb, n=1)
            metric_logger.meters["r2score_atac"].update(r2score_motif_emb, n=1)
            metric_logger.meters["pearsonr_score_atac"].update(pearsonr_score_motif_emb, n=1)
            metric_logger.meters["spearmanr_score_atac"].update(spearmanr_score_motif_emb, n=1)
            print('Done')
            if printlog:
                print(
                    "* Score@R2 {r2:.3f} Score@pearsonr {pearson:.3f} Score@spearmanr {spearman:.3f}  loss_seq_emb {loss_seq_emb:.3f}\n Score@R2 {r2score_motif_aprofile:.3f} Score@pearsonr {pearsonr_score_motif_aprofile:.3f} Score@spearmanr {spearmanr_score_motif_aprofile:.3f}  loss_motif_aprofile {loss_motif_emb:.3f} loss_latent_embed {loss_latent_emb:.3f}".format(
                        r2=r2score_seq_emb,
                        pearson=pearsonr_score_seq_emb,
                        spearman=spearmanr_score_seq_emb,
                        r2score_motif_aprofile=r2score_motif_emb,
                        pearsonr_score_motif_aprofile=pearsonr_score_motif_emb,
                        spearmanr_score_motif_aprofile=spearmanr_score_motif_emb,
                        loss_seq_emb=loss_seq_emb,
                        loss_motif_emb=loss_motif_emb,
                        loss_latent_emb = loss_latent_emb,
                    )
                )
        else: 
            """If train_motif_only is False, train for both motif_embeddings and sequence_embeddings"""
            bin=100
            print("1/3 Compress motif_embeddings")
            preds_motif_emb = np.concatenate(preds_motif_emb, axis=0)
            num_complete_bins = preds_motif_emb.size // bin
            if num_complete_bins > 0:
                complete_bins = preds_motif_emb[:num_complete_bins * bin].reshape(-1, bin)
                complete_bin_means = complete_bins.mean(axis=1)
            if preds_motif_emb.size % bin > 0:
                remainder = preds_motif_emb[num_complete_bins * bin:]
                remainder_mean = np.array([remainder.mean()])  
                preds_motif_aprofile = np.concatenate([complete_bin_means, remainder_mean])
            else:
                preds_motif_aprofile = complete_bin_means

            obs_motif_emb = np.concatenate(obs_motif_emb, axis=0)
            num_complete_bins = obs_motif_emb.size // bin
            if num_complete_bins > 0:
                complete_bins = obs_motif_emb[:num_complete_bins * bin].reshape(-1, bin)
                complete_bin_means = complete_bins.mean(axis=1)
            if obs_motif_emb.size % bin > 0:
                remainder = obs_motif_emb[num_complete_bins * bin:]
                remainder_mean = np.array([remainder.mean()])  
                obs_motif_aprofile = np.concatenate([complete_bin_means, remainder_mean])
            else:
                obs_motif_aprofile = complete_bin_means
            print('2/3 Calculating motif embedding metric scores')
            print(f'preds_motif_emb.shape: {preds_motif_aprofile.shape}')
            r2score_motif_emb, pearsonr_score_motif_emb, spearmanr_score_motif_emb = cal_score_stats(preds_motif_aprofile, obs_motif_aprofile, data_loader, args)
            print('3/3 Updating metric scores')
            metric_logger.meters["r2score_atac"].update(r2score_motif_emb, n=1)
            metric_logger.meters["pearsonr_score_atac"].update(pearsonr_score_motif_emb, n=1)
            metric_logger.meters["spearmanr_score_atac"].update(spearmanr_score_motif_emb, n=1)
            metric_logger.meters["r2score"].update(0.00, n=1)
            metric_logger.meters["pearsonr_score"].update(0.00, n=1)    
            metric_logger.meters["spearmanr_score"].update(0.00, n=1)
            print('Done')
            if printlog:
                print(
                    "* Score@R2 {r2score_motif_aprofile:.3f} Score@pearsonr {pearsonr_score_motif_aprofile:.3f} Score@spearmanr {spearmanr_score_motif_aprofile:.3f}  loss_motif_aprofile {loss_motif_emb:.3f} loss_latent_embed {loss_latent_emb:.3f}".format(
                        r2score_motif_aprofile=r2score_motif_emb,
                        pearsonr_score_motif_aprofile=pearsonr_score_motif_emb,
                        spearmanr_score_motif_aprofile=spearmanr_score_motif_emb,
                        loss_motif_emb=loss_motif_emb,
                        loss_latent_emb = loss_latent_emb,
                    )
                )
    else:
        preds = np.concatenate(preds, axis=0).reshape(-1)
        obs = np.concatenate(obs, axis=0).reshape(-1)
        preds_atac = np.concatenate(preds_atac, axis=0).reshape(-1)
        obs_atac = np.concatenate(obs_atac, axis=0).reshape(-1)

        r2score, pearsonr_score, spearmanr_score = cal_score_stats(preds, obs, data_loader, args)
        r2score_atac, pearsonr_score_atac, spearmanr_score_atac = cal_score_stats(preds_atac, obs_atac, data_loader, args)

        metric_logger.meters["r2score"].update(r2score, n=1)
        metric_logger.meters["pearsonr_score"].update(pearsonr_score, n=1)
        metric_logger.meters["spearmanr_score"].update(spearmanr_score, n=1)

        metric_logger.meters["r2score_atac"].update(r2score_atac, n=1)
        metric_logger.meters["pearsonr_score_atac"].update(pearsonr_score_atac, n=1)
        metric_logger.meters["spearmanr_score_atac"].update(spearmanr_score_atac, n=1)

        if printlog:
            print(
                "* Score@R2 {r2:.3f} Score@pearsonr {pearson:.3f} Score@spearmanr {spearman:.3f}  loss {losses.global_avg:.3f}".format(
                    r2=r2score,
                    pearson=pearsonr_score,
                    spearman=spearmanr_score,
                    r2_atac=r2score_atac,
                    pearson_atac=pearsonr_score_atac,
                    spearman_atac=spearmanr_score_atac,
                    losses=metric_logger.loss,
                )
            )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

