#!/usr/bin/env python
"""Reproduce rs55705857-to-MYC motif scoring across GET checkpoints.

The historical analysis had two distinct stages:

1. Run a checkpoint in ``interpret`` mode to save MYC input Jacobians.
2. Score each motif as its alternate-minus-reference sequence change multiplied
   by the checkpoint-specific MYC motif Jacobian.

Use the ``interpret`` subcommand once per checkpoint/condition, then pass the
resulting Zarr stores to ``score``.  The score output contains both the raw
current score and the thresholded score used by the November 2023 glioma
analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_VARIANT = "rs55705857"
DEFAULT_GENE = "MYC"


def _parse_condition(value: str) -> tuple[str, Path]:
    """Parse LABEL=/path/to/interpretation.zarr."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Conditions must use LABEL=/path/to/interpretation.zarr"
        )
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "Conditions must use LABEL=/path/to/interpretation.zarr"
        )
    return label.strip(), Path(raw_path).expanduser()


def _parse_legacy_condition(value: str) -> tuple[str, str]:
    """Parse LABEL=legacy_cell_id."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("Legacy conditions must use LABEL=CELL_ID")
    label, cell_id = value.split("=", 1)
    if not label.strip() or not cell_id.strip():
        raise argparse.ArgumentTypeError("Legacy conditions must use LABEL=CELL_ID")
    return label.strip(), cell_id.strip()


def _set_nested(cfg, key: str, value) -> None:
    """Update a possibly structured OmegaConf object."""
    from omegaconf import OmegaConf

    OmegaConf.update(cfg, key, value, merge=False, force_add=True)


def run_interpret(args: argparse.Namespace) -> Path:
    """Generate a MYC interpretation Zarr from one fine-tuned checkpoint."""
    from omegaconf import OmegaConf

    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Resolved GET config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = OmegaConf.load(config_path)
    output_dir = args.output_dir.expanduser().resolve()
    project_name = args.project_name
    run_name = args.label

    overrides = {
        "stage": "predict",
        "run.project_name": project_name,
        "run.run_name": run_name,
        "run.use_wandb": False,
        "machine.output_dir": str(output_dir),
        "dataset.celltypes": args.celltype,
        "dataset.leave_out_celltypes": args.celltype,
        "dataset.leave_out_chromosomes": None,
        "task.test_mode": "interpret",
        "task.gene_list": args.gene,
        # Empty layer_names still records input/region_motif gradients and avoids
        # retaining unrelated intermediate-layer Jacobians.
        "task.layer_names": [],
        "finetune.checkpoint": str(checkpoint_path),
        "finetune.resume_ckpt": None,
    }
    if args.num_devices is not None:
        overrides["machine.num_devices"] = args.num_devices
    if args.num_workers is not None:
        overrides["machine.num_workers"] = args.num_workers
    if args.batch_size is not None:
        overrides["machine.batch_size"] = args.batch_size

    if args.checkpoint_format == "lightning":
        overrides["finetune.model_key"] = "state_dict"
        overrides["finetune.rename_config"] = {"model.": ""}
    elif args.checkpoint_format == "model":
        overrides["finetune.model_key"] = "model"
        overrides["finetune.rename_config"] = None

    for key, value in overrides.items():
        _set_nested(cfg, key, value)

    run_dir = output_dir / project_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "effective_interpret_config.yaml")

    if args.backend == "region-zarr":
        from get_model.run_region import run_zarr

        run_zarr(cfg)
    else:
        from get_model.run_region import run

        run(cfg)

    zarr_path = run_dir / f"{args.celltype}.zarr"
    if not zarr_path.exists():
        raise RuntimeError(
            f"Interpretation finished but expected Zarr was not created: {zarr_path}"
        )
    print(zarr_path)
    return zarr_path


def _load_genome(genome_fasta: Path | None):
    from gcell.dna.genome import Genome

    if genome_fasta is None:
        return Genome("hg38")

    from pyfaidx import Fasta

    fasta_path = genome_fasta.expanduser().resolve()
    if not fasta_path.is_file():
        raise FileNotFoundError(f"Genome FASTA not found: {fasta_path}")
    genome = Genome("hg38", load_genome_seq=False)
    genome.genome_seq = Fasta(str(fasta_path))
    genome.chr_suffix = (
        "chr" if list(genome.genome_seq.keys())[0].startswith("chr") else ""
    )
    return genome


def _load_variant(args: argparse.Namespace, genome):
    from gcell.dna.mutation import Mutations, read_rsid_parallel

    if args.variant_table is not None:
        table_path = args.variant_table.expanduser().resolve()
        variants = pd.read_csv(table_path, sep="\t")
        required = {"Chromosome", "Start", "End", "Ref", "Alt", "RSID"}
        missing = sorted(required - set(variants.columns))
        if missing:
            raise ValueError(
                f"Variant table {table_path} is missing columns: {', '.join(missing)}"
            )
        variants = variants.loc[variants["RSID"] == args.variant, sorted(required)]
        if len(variants) != 1:
            raise ValueError(
                f"Expected exactly one {args.variant} row in {table_path}; found {len(variants)}"
            )
        mutation = Mutations(genome, variants)
    else:
        mutation, processed, failed = read_rsid_parallel(
            genome, [args.variant], num_workers=1
        )
        if args.variant not in processed or args.variant in failed or mutation.df.empty:
            raise RuntimeError(
                f"Could not resolve {args.variant} through Ensembl. Supply --variant-table."
            )
        # Ensembl can return secondary mappings; freeze one canonical GRCh38 row.
        variant_rows = mutation.df.loc[mutation.df["RSID"] == args.variant].copy()
        if len(variant_rows) != 1:
            raise ValueError(
                f"Ensembl returned {len(variant_rows)} GRCh38 mappings for {args.variant}; "
                "supply --variant-table to select the intended allele."
            )
        mutation = Mutations(
            genome,
            variant_rows[["Chromosome", "Start", "End", "Ref", "Alt", "RSID"]],
        )

    row = mutation.df.iloc[0]
    if len(row.Ref) != 1 or len(row.Alt) != 1:
        raise ValueError("This reproduction script currently supports SNVs only")
    ref_sequence = str(row.Ref_seq)
    center_base = ref_sequence[len(ref_sequence) // 2].upper()
    if center_base != str(row.Ref).upper():
        raise ValueError(
            f"Reference-allele mismatch at {row.Chromosome}:{row.End}: "
            f"FASTA={center_base}, table={row.Ref}. Check assembly and coordinates."
        )
    return mutation


def _motif_delta(mutation, motif) -> pd.Series:
    motif_diff = mutation.get_motif_diff(motif)
    delta = motif_diff["Alt"].iloc[0] - motif_diff["Ref"].iloc[0]
    delta.name = "motif_delta_raw"
    return delta.astype(float)


def _legacy_threshold(delta: pd.Series) -> pd.Series:
    """Reproduce the loss/gain discretization in the 2023 glioma script."""
    return pd.Series(
        np.where(delta < -10, -1.0, np.where(delta > 10, 1.0, 0.0)),
        index=delta.index,
        name="motif_delta_legacy",
    )


def _score_hydra_condition(
    label: str,
    zarr_path: Path,
    gene: str,
    delta_raw: pd.Series,
) -> pd.DataFrame:
    from gcell.cell.celltype import GETHydraCellType

    if not zarr_path.exists():
        raise FileNotFoundError(f"Interpretation Zarr not found: {zarr_path}")
    cell = GETHydraCellType(
        celltype=label,
        zarr_path=str(zarr_path),
        prediction_target="exp",
    )
    if gene not in set(cell.gene_annot["gene_name"]):
        raise ValueError(f"{gene} is not present in {label} interpretation Zarr")

    importance = cell.get_gene_jacobian_summary(gene, axis="motif").astype(float)
    motifs = delta_raw.index.intersection(importance.index, sort=False)
    if motifs.empty:
        raise ValueError(
            f"No shared motif names between the sequence scan and {label} Jacobians"
        )

    result = pd.DataFrame(
        {
            "condition": label,
            "motif": motifs,
            "motif_delta_raw": delta_raw.loc[motifs].values,
            "motif_importance": importance.loc[motifs].values,
        }
    )
    result["motif_delta_legacy"] = _legacy_threshold(
        result.set_index("motif")["motif_delta_raw"]
    ).values
    result["score_raw"] = result["motif_delta_raw"] * result["motif_importance"]
    result["score_legacy"] = result["motif_delta_legacy"] * result["motif_importance"]
    result["rank_abs_raw"] = (
        result["score_raw"].abs().rank(method="min", ascending=False).astype(int)
    )
    result["rank_abs_legacy"] = (
        result["score_legacy"].abs().rank(method="min", ascending=False).astype(int)
    )
    return result


def _prepare_legacy_input_cache(cell_id: str, data_dir: Path, cache_dir: Path) -> Path:
    """Link large legacy inputs into a writable cache for gene annotations."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("csv", "watac.npz"):
        source = (data_dir / f"{cell_id}.{suffix}").resolve()
        destination = cache_dir / source.name
        if not source.is_file():
            raise FileNotFoundError(f"Legacy healthy-cell input not found: {source}")
        if not destination.exists():
            destination.symlink_to(source)
    return cache_dir


def _score_legacy_condition(
    label: str,
    cell_id: str,
    data_dir: Path,
    interpret_dir: Path,
    cache_dir: Path,
    gene: str,
    delta_raw: pd.Series,
) -> pd.DataFrame:
    """Score one legacy healthy-cell interpretation store."""
    from gcell.cell.celltype import GETCellType
    from gcell.cell.mutincell import _prepare_celltype_config
    from omegaconf import OmegaConf

    writable_data_dir = _prepare_legacy_input_cache(cell_id, data_dir, cache_dir)
    cfg = _prepare_celltype_config(
        OmegaConf.create(
            {
                "s3_file_sys": None,
                "celltype": {
                    "features": "NrMotifV1",
                    "num_region_per_sample": 200,
                    "data_dir": f"{writable_data_dir}/",
                    "interpret_dir": str(interpret_dir),
                    "input": True,
                    "jacob": True,
                    "embed": False,
                    "assets_dir": "",
                    "num_cls": 2,
                },
            }
        )
    )
    cell = GETCellType(cell_id, cfg)
    importance = cell.get_gene_jacobian_summary(gene, axis="motif").astype(float)
    motifs = delta_raw.index.intersection(importance.index, sort=False)
    if motifs.empty:
        raise ValueError(
            f"No shared motif names between the sequence scan and legacy cell {cell_id}"
        )
    result = pd.DataFrame(
        {
            "condition": label,
            "motif": motifs,
            "motif_delta_raw": delta_raw.loc[motifs].values,
            "motif_importance": importance.loc[motifs].values,
        }
    )
    result["motif_delta_legacy"] = _legacy_threshold(
        result.set_index("motif")["motif_delta_raw"]
    ).values
    result["score_raw"] = result["motif_delta_raw"] * result["motif_importance"]
    result["score_legacy"] = result["motif_delta_legacy"] * result["motif_importance"]
    result["rank_abs_raw"] = (
        result["score_raw"].abs().rank(method="min", ascending=False).astype(int)
    )
    result["rank_abs_legacy"] = (
        result["score_legacy"].abs().rank(method="min", ascending=False).astype(int)
    )
    return result


def run_score(args: argparse.Namespace) -> Path:
    """Score rs55705857 motif effects for one or more interpretation Zarrs."""
    hydra_conditions = dict(args.condition or [])
    legacy_conditions = dict(args.legacy_condition or [])
    all_labels = list(hydra_conditions) + list(legacy_conditions)
    if not all_labels:
        raise ValueError("Supply at least one --condition or --legacy-condition")
    if len(set(all_labels)) != len(all_labels):
        raise ValueError("Condition labels must be unique")
    if legacy_conditions and (
        args.legacy_data_dir is None or args.legacy_interpret_dir is None
    ):
        raise ValueError(
            "Legacy conditions require --legacy-data-dir and --legacy-interpret-dir"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    genome = _load_genome(args.genome_fasta)
    mutation = _load_variant(args, genome)

    from gcell.dna.nr_motif_v1 import NrMotifV1

    motif = (
        NrMotifV1.load_from_pickle()
        if args.motif_pickle is None
        else NrMotifV1.load_from_pickle(args.motif_pickle.expanduser().resolve())
    )
    delta_raw = _motif_delta(mutation, motif)

    score_tables = [
        _score_hydra_condition(label, path.expanduser().resolve(), args.gene, delta_raw)
        for label, path in hydra_conditions.items()
    ]
    if legacy_conditions:
        legacy_cache_dir = (
            args.legacy_cache_dir.expanduser().resolve()
            if args.legacy_cache_dir is not None
            else output_dir / "legacy_input_cache"
        )
        score_tables.extend(
            _score_legacy_condition(
                label=label,
                cell_id=cell_id,
                data_dir=args.legacy_data_dir.expanduser().resolve(),
                interpret_dir=args.legacy_interpret_dir.expanduser().resolve(),
                cache_dir=legacy_cache_dir / cell_id,
                gene=args.gene,
                delta_raw=delta_raw,
            )
            for label, cell_id in legacy_conditions.items()
        )
    scores = pd.concat(score_tables, ignore_index=True)
    scores.insert(0, "gene", args.gene)
    scores.insert(0, "variant", args.variant)

    stem = f"{args.variant}_{args.gene}"
    score_path = output_dir / f"{stem}_motif_scores.tsv"
    scores.to_csv(score_path, sep="\t", index=False)

    rank_column = "rank_abs_legacy" if args.rank_by == "legacy" else "rank_abs_raw"
    score_column = "score_legacy" if args.rank_by == "legacy" else "score_raw"
    top = (
        scores.loc[scores[score_column] != 0]
        .assign(_abs_score=lambda frame: frame[score_column].abs())
        .sort_values(
            ["condition", "_abs_score", "motif"], ascending=[True, False, True]
        )
        .groupby("condition", sort=False)
        .head(args.top_n)
        .drop(columns="_abs_score")
    )
    top.to_csv(output_dir / f"{stem}_top{args.top_n}.tsv", sep="\t", index=False)

    highlighted = scores.loc[
        scores["motif"].str.contains(args.highlight_regex, case=False, regex=True)
    ].sort_values(["condition", rank_column, "motif"])
    highlighted.to_csv(
        output_dir / f"{stem}_MYC_OCT_POU_motifs.tsv", sep="\t", index=False
    )

    comparison = scores.pivot(
        index="motif",
        columns="condition",
        values=["motif_importance", "score_raw", "score_legacy"],
    )
    comparison.columns = [f"{metric}__{condition}" for metric, condition in comparison]
    comparison.reset_index().to_csv(
        output_dir / f"{stem}_condition_comparison.tsv", sep="\t", index=False
    )

    variant_row = mutation.df.iloc[0]
    manifest = {
        "variant": args.variant,
        "gene": args.gene,
        "assembly": "hg38",
        "coordinates": {
            "chromosome": str(variant_row.Chromosome),
            "start_0_based": int(variant_row.Start),
            "end_0_based_exclusive": int(variant_row.End),
            "ref": str(variant_row.Ref),
            "alt": str(variant_row.Alt),
        },
        "conditions": {
            **{
                label: {"type": "hydra_zarr", "path": str(path.resolve())}
                for label, path in hydra_conditions.items()
            },
            **{
                label: {
                    "type": "legacy_healthy_cell",
                    "cell_id": cell_id,
                    "data_dir": str(args.legacy_data_dir.resolve()),
                    "interpret_dir": str(args.legacy_interpret_dir.resolve()),
                }
                for label, cell_id in legacy_conditions.items()
            },
        },
        "score_definitions": {
            "raw": "(Alt motif score - Ref motif score) * MYC motif Jacobian",
            "legacy": "sign(Alt-Ref) when abs(Alt-Ref)>10, else 0; multiplied by MYC motif Jacobian",
        },
        "rank_by": args.rank_by,
    }
    (output_dir / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(top.to_string(index=False))
    print(f"\nFull scores: {score_path}")
    return score_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    interpret = subparsers.add_parser(
        "interpret", help="generate checkpoint-specific MYC interpretation Zarr"
    )
    interpret.add_argument("--config", type=Path, required=True)
    interpret.add_argument("--checkpoint", type=Path, required=True)
    interpret.add_argument("--label", required=True, help="condition/run label")
    interpret.add_argument("--celltype", default="tumor")
    interpret.add_argument("--gene", default=DEFAULT_GENE)
    interpret.add_argument("--output-dir", type=Path, required=True)
    interpret.add_argument("--project-name", default="rs55705857_MYC_interpret")
    interpret.add_argument(
        "--backend", choices=("region-zarr", "region"), default="region-zarr"
    )
    interpret.add_argument(
        "--checkpoint-format",
        choices=("lightning", "model", "config"),
        default="lightning",
        help="how weights are nested/named; 'config' preserves the YAML settings",
    )
    interpret.add_argument("--num-devices", type=int)
    interpret.add_argument("--num-workers", type=int)
    interpret.add_argument("--batch-size", type=int)
    interpret.set_defaults(func=run_interpret)

    score = subparsers.add_parser(
        "score", help="score motif effects from one or more interpretation Zarrs"
    )
    score.add_argument(
        "--condition",
        action="append",
        type=_parse_condition,
        metavar="LABEL=ZARR",
        help="repeat for IDHwt, IDHmut, and optional healthy conditions",
    )
    score.add_argument(
        "--legacy-condition",
        action="append",
        type=_parse_legacy_condition,
        metavar="LABEL=CELL_ID",
        help="repeat for saved legacy healthy-cell stores, e.g. healthy_OPC=4",
    )
    score.add_argument("--legacy-data-dir", type=Path)
    score.add_argument("--legacy-interpret-dir", type=Path)
    score.add_argument(
        "--legacy-cache-dir",
        type=Path,
        help="writable cache for generated legacy gene annotations",
    )
    score.add_argument("--variant", default=DEFAULT_VARIANT)
    score.add_argument("--gene", default=DEFAULT_GENE)
    score.add_argument("--variant-table", type=Path)
    score.add_argument("--genome-fasta", type=Path)
    score.add_argument("--motif-pickle", type=Path)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--top-n", type=int, default=20)
    score.add_argument("--rank-by", choices=("legacy", "raw"), default="legacy")
    score.add_argument("--highlight-regex", default=r"MYC|OCT|POU")
    score.set_defaults(func=run_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
