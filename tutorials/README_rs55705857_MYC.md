# rs55705857 to MYC variant scoring

This analysis requires `gcell` commit `5a2fac9` or later. That commit removes
the legacy account-specific path overrides from `CellMutCollection` and fixes
explicit FASTA loading.

`reproduce_rs55705857_myc.py` separates model interpretation from variant
scoring. `CellMutCollection` does not consume a checkpoint directly: a GET
checkpoint must first generate a tumor-cell interpretation Zarr containing the
MYC input Jacobian.

The November 2023 source is still available in Git commit
`5607bf583fa89421701f7360effcf9d20db8cd28` as `analysis/glioma_orig.py`; it was
removed from the working tree by commit `04101a5d2a3926842ea65d7b2ba02ea0085e4fa9`.

Use the resolved training config for each model (normally the saved Hydra
`config.yaml`) so the model and dataset definitions match the checkpoint.

```bash
python tutorials/reproduce_rs55705857_myc.py interpret \
  --config /path/to/IDHwt/resolved_config.yaml \
  --checkpoint /insomnia001/depts/pmg/users/jpd2207/GBM_get_pj/finetune/finetune_GBM_idhwt_6sp_3ct_08_21/finetune_GBM_idhwt_6sp_3ct_08_21_chr21_full30/checkpoints/best.ckpt \
  --checkpoint-format lightning \
  --label IDHwt_n6 \
  --celltype tumor \
  --output-dir /path/to/rs55705857_results
```

Run the same command with the IDH-mutant config/checkpoint and
`--label IDHmut_n1`. The expected outputs are:

```text
/path/to/rs55705857_results/rs55705857_MYC_interpret/IDHwt_n6/tumor.zarr
/path/to/rs55705857_results/rs55705857_MYC_interpret/IDHmut_n1/tumor.zarr
```

Then score and compare both checkpoints:

```bash
python tutorials/reproduce_rs55705857_myc.py score \
  --condition IDHwt_n6=/path/to/rs55705857_results/rs55705857_MYC_interpret/IDHwt_n6/tumor.zarr \
  --condition IDHmut_n1=/path/to/rs55705857_results/rs55705857_MYC_interpret/IDHmut_n1/tumor.zarr \
  --variant-table tutorials/rs55705857.hg38.tsv \
  --genome-fasta /path/to/hg38.fa \
  --output-dir /path/to/rs55705857_results/scores
```

The full output retains both the continuous score and the explicitly labeled
`thresholded_2023` score. The continuous score is `(Alt - Ref motif score) *
MYC motif Jacobian`. The `thresholded_2023` calculation first maps motif changes
below -10 to -1, above 10 to +1, and all other changes to zero.

The checked healthy reference results in
`rs55705857_MYC_healthy_reference/` were generated from saved legacy stores:

- healthy OPC: cell ID 4
- healthy oligodendrocyte: cell ID 38
- legacy data: `/mnt/storage/get_demo/pretrain_human_bingren_shendure_apr2023/fetal_adult`
- legacy interpretations: `/mnt/storage/get_demo/Interpretation_all_hg38_allembed_v4_natac`

They can be regenerated with:

```bash
python tutorials/reproduce_rs55705857_myc.py score \
  --legacy-condition healthy_OPC=4 \
  --legacy-condition healthy_oligodendrocyte=38 \
  --legacy-data-dir /mnt/storage/get_demo/pretrain_human_bingren_shendure_apr2023/fetal_adult \
  --legacy-interpret-dir /mnt/storage/get_demo/Interpretation_all_hg38_allembed_v4_natac \
  --variant-table tutorials/rs55705857.hg38.tsv \
  --genome-fasta /home/xf2217/.gcell_data/genomes/hg38.fa \
  --output-dir tutorials/rs55705857_MYC_healthy_reference
```
