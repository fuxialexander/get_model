from get_model.model.model import ATACBERT 
from get_model.model.trainer import data_distributed_parallel_gpu
import torch.multiprocessing as mp
import sys
sys.path.append('/share/vault/Users/gz2294/get_model')
from get_model.dataset.zarr_dataset import ATACBERTDataset


if __name__ == '__main__':
    pretrain = ATACBERTDataset(['/share/vault/Users/gz2294/get_data/encode_hg38atac_dense.zarr',],
                            '/share/vault/Users/gz2294/get_data/hg38.zarr', 
                            '/share/vault/Users/gz2294/get_data/hg38_motif_result.zarr', 
                            ['/share/vault/Users/gz2294/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', 
                            '/share/vault/Users/gz2294/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], 
                            peak_name='peaks_q0.01_tissue_open_exp', 
                            preload_count=100, n_packs=1,
                            max_peak_length=5000, 
                            center_expand_target=500, 
                            n_peaks_lower_bound=1, 
                            n_peaks_upper_bound=1, use_insulation=False, 
                            leave_out_celltypes=None, leave_out_chromosomes='chr1', 
                            is_train=False, dataset_size=65536, 
                            additional_peak_columns=['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'], 
                            hic_path=None)
    model = ATACBERT(
            num_layers = 32,
            embed_dim = 512,
            attention_heads = 16,
            token_dropout = True
        )
    mp.spawn(
        data_distributed_parallel_gpu,
        args=(model, pretrain, 4, 20, 1000, 1),
        nprocs=4,
        join=True)