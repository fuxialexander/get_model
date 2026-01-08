"""
Mutation Effect Prediction and Benchmarking for ATAC Signal Models

Evaluates model's ability to predict chromatin accessibility changes
from genetic variants using caQTL benchmarking data.

The evaluation:
1. Loads variants from caQTL benchmarking TSV
2. For each variant, creates ref and alt sequences
3. Predicts ATAC signal for both sequences
4. Computes log fold change of summed profiles
5. Compares predicted effects to observed effects
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, Optional
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Nucleotide to one-hot index mapping
NUCLEOTIDE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}


def load_caqtl_variants(path: str, filter_used: bool = True) -> pd.DataFrame:
    """
    Load caQTL variants from TSV file.

    Args:
        path: Path to the TSV file (gzipped or plain)
        filter_used: If True, only return variants with var.isused == True

    Returns:
        DataFrame with variant information
    """
    # Try to detect if file is gzipped or plain TSV
    if path.endswith('.gz'):
        # Try plain TSV first (unzipped version)
        plain_path = path[:-3]  # Remove .gz extension
        if os.path.exists(plain_path):
            df = pd.read_csv(plain_path, sep='\t')
        else:
            # Try as gzip
            try:
                df = pd.read_csv(path, sep='\t', compression='gzip')
            except Exception:
                # Might be a zip file renamed to .gz
                import zipfile
                with zipfile.ZipFile(path, 'r') as zf:
                    name = zf.namelist()[0]
                    with zf.open(name) as f:
                        df = pd.read_csv(f, sep='\t')
    else:
        df = pd.read_csv(path, sep='\t')

    # Filter to used variants if requested
    if filter_used and 'var.isused' in df.columns:
        df = df[df['var.isused'] == True].copy()

    print(f"Loaded {len(df)} caQTL variants from {path}")
    return df


class MutationEffectEvaluator:
    """
    Evaluates model mutation effect predictions against caQTL data.

    This class:
    1. Loads caQTL variant data
    2. Generates ref/alt sequence pairs for each variant
    3. Runs model predictions on both sequences
    4. Computes log fold change of predicted signals
    5. Compares to observed effects (obs.meanLog2FC)

    Metrics computed:
    - Pearson/Spearman correlation with observed effects
    - AUC-ROC for predicting effect direction (up vs down)
    - AUC-ROC for predicting strong effects (|effect| > 1)
    """

    def __init__(
        self,
        genome_io,
        variants_path: str,
        extend_bp: int = 1024,
        center_crop: int = 1024,
        filter_used: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            genome_io: DenseZarrIO or similar object for genome sequence access
            variants_path: Path to caQTL TSV file
            extend_bp: Base pairs to extend from variant center (total = 2 * extend_bp)
            center_crop: Base pairs to use for effect calculation (center region)
            filter_used: Whether to filter to var.isused == True variants
        """
        self.genome_io = genome_io
        self.extend_bp = extend_bp
        self.center_crop = center_crop

        # Load variants
        self.variants = load_caqtl_variants(variants_path, filter_used=filter_used)

    def get_ref_alt_sequences(
        self, chrom: str, pos: int, ref: str, alt: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get reference and alternative sequences for a variant.

        Args:
            chrom: Chromosome (e.g., 'chr1')
            pos: Position in GRCh38 (1-based)
            ref: Reference allele
            alt: Alternative allele

        Returns:
            Tuple of (ref_seq, alt_seq), each shape (seq_len, 4)
        """
        # Convert 1-based position to 0-based for slicing
        pos_0based = pos - 1
        start = pos_0based - self.extend_bp
        end = pos_0based + self.extend_bp

        # Ensure non-negative start
        if start < 0:
            raise ValueError(f"Variant at {chrom}:{pos} too close to chromosome start")

        # Get reference sequence from genome
        ref_seq = self.genome_io.get_track(chrom, start, end, output_format="raw_array")

        # Handle different output formats from genome_io
        if ref_seq.shape[0] == 4:
            ref_seq = ref_seq.T  # Convert from (4, length) to (length, 4)

        # Verify we got the expected length
        expected_len = 2 * self.extend_bp
        if ref_seq.shape[0] != expected_len:
            raise ValueError(
                f"Got sequence length {ref_seq.shape[0]}, expected {expected_len}"
            )

        # The variant is at the center of the sequence
        var_idx = self.extend_bp

        # Create alternative sequence by modifying the variant position
        alt_seq = ref_seq.copy()

        # Handle SNPs only for now (single nucleotide variants)
        if len(ref) == 1 and len(alt) == 1:
            # Set variant position to alternative allele
            alt_seq[var_idx, :] = 0
            if alt.upper() in NUCLEOTIDE_TO_IDX:
                alt_seq[var_idx, NUCLEOTIDE_TO_IDX[alt.upper()]] = 1
            else:
                raise ValueError(f"Unknown nucleotide: {alt}")
        else:
            # For indels, skip for now
            raise ValueError(f"Indels not supported: {ref} -> {alt}")

        return ref_seq, alt_seq

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        device: torch.device,
        desc: str = "caQTL Evaluation",
        use_amp: bool = False,
        max_variants: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run mutation effect evaluation on all variants.

        Args:
            model: The trained model (should accept sequence input and return ATAC predictions)
            device: Device to run inference on
            desc: Description for progress bar
            use_amp: Use automatic mixed precision
            max_variants: Maximum number of variants to evaluate (for faster testing)

        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()

        predicted_logfc = []
        observed_logfc = []
        variant_ids = []
        skipped = 0

        variants_to_eval = self.variants
        if max_variants is not None:
            variants_to_eval = variants_to_eval.head(max_variants)

        for _, row in tqdm(variants_to_eval.iterrows(), total=len(variants_to_eval),
                           desc=desc, leave=False):
            try:
                # Get chromosome and position columns
                chrom = row.get('var.chr', row.get('chr', None))
                pos = row.get('var.pos_hg38', row.get('pos', None))
                ref_allele = row.get('allele1', row.get('ref', None))
                alt_allele = row.get('allele2', row.get('alt', None))

                if chrom is None or pos is None:
                    skipped += 1
                    continue

                # Get ref and alt sequences
                ref_seq, alt_seq = self.get_ref_alt_sequences(
                    chrom,
                    int(pos),
                    ref_allele,
                    alt_allele
                )

                # Convert to tensors and add batch dimension
                # Shape: (batch=1, seq_len, 4)
                ref_tensor = torch.tensor(
                    ref_seq, dtype=torch.float32
                ).unsqueeze(0).to(device)
                alt_tensor = torch.tensor(
                    alt_seq, dtype=torch.float32
                ).unsqueeze(0).to(device)

                # Run model predictions
                with torch.no_grad():
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            ref_output = model(ref_tensor)
                            alt_output = model(alt_tensor)
                    else:
                        ref_output = model(ref_tensor)
                        alt_output = model(alt_tensor)

                # Extract ATAC predictions
                if isinstance(ref_output, dict):
                    ref_pred = ref_output.get('atac_prediction', ref_output.get('atac', None))
                    alt_pred = alt_output.get('atac_prediction', alt_output.get('atac', None))
                    if ref_pred is None:
                        # Try to get the first tensor value
                        ref_pred = list(ref_output.values())[0]
                        alt_pred = list(alt_output.values())[0]
                else:
                    ref_pred = ref_output
                    alt_pred = alt_output

                ref_pred = ref_pred.squeeze()
                alt_pred = alt_pred.squeeze()

                # Center crop for effect calculation
                seq_len = ref_pred.shape[0]
                crop_start = (seq_len - self.center_crop) // 2
                crop_end = crop_start + self.center_crop

                # Sum over center region
                ref_sum = ref_pred[crop_start:crop_end].sum().item()
                alt_sum = alt_pred[crop_start:crop_end].sum().item()

                # Compute log fold change (alt / ref)
                # Adding pseudocount to avoid log(0)
                log_fc = np.log2((alt_sum + 1) / (ref_sum + 1))

                predicted_logfc.append(log_fc)
                obs_value = row.get('obs.meanLog2FC', row.get('log2FC', 0))
                observed_logfc.append(float(obs_value))
                variant_ids.append(row.get('var.snp_id', row.get('snp_id', f'var_{len(variant_ids)}')))

            except Exception as e:
                skipped += 1
                continue

        # Convert to arrays
        pred = np.array(predicted_logfc)
        obs = np.array(observed_logfc)

        if len(pred) == 0:
            print(f"Warning: No variants successfully evaluated (skipped {skipped})")
            return {
                'n_variants': 0,
                'n_skipped': skipped,
                'pearson_r': float('nan'),
                'spearman_r': float('nan'),
                'auc_direction': float('nan'),
                'auc_strong_effects': float('nan'),
            }

        # Compute correlation metrics
        pearson_r, pearson_p = stats.pearsonr(pred, obs)
        spearman_r, spearman_p = stats.spearmanr(pred, obs)

        # Compute classification metrics (direction prediction)
        obs_direction = (obs > 0).astype(int)
        pred_direction_score = pred

        try:
            auc_direction = roc_auc_score(obs_direction, pred_direction_score)
        except ValueError:
            auc_direction = float('nan')

        # Compute AUC for strong effects (|obs| > 1)
        strong_mask = np.abs(obs) > 1
        if strong_mask.sum() >= 10:
            obs_strong = (obs[strong_mask] > 0).astype(int)
            pred_strong = pred[strong_mask]
            try:
                auc_strong = roc_auc_score(obs_strong, pred_strong)
            except ValueError:
                auc_strong = float('nan')
        else:
            auc_strong = float('nan')

        return {
            'n_variants': len(pred),
            'n_skipped': skipped,
            'pearson_r': float(pearson_r),
            'spearman_r': float(spearman_r),
            'auc_direction': float(auc_direction),
            'auc_strong_effects': float(auc_strong),
            'predicted_logfc': pred,
            'observed_logfc': obs,
        }

    def get_detailed_predictions(
        self,
        model: torch.nn.Module,
        device: torch.device,
        use_amp: bool = False,
    ) -> pd.DataFrame:
        """
        Get detailed predictions for all variants.

        Returns a DataFrame with variant info and predicted effects.
        Useful for downstream analysis and visualization.
        """
        model.eval()

        results = []

        for _, row in tqdm(self.variants.iterrows(), total=len(self.variants),
                           desc="Getting predictions", leave=False):
            try:
                chrom = row.get('var.chr', row.get('chr', None))
                pos = row.get('var.pos_hg38', row.get('pos', None))
                ref_allele = row.get('allele1', row.get('ref', None))
                alt_allele = row.get('allele2', row.get('alt', None))

                if chrom is None or pos is None:
                    continue

                ref_seq, alt_seq = self.get_ref_alt_sequences(
                    chrom, int(pos), ref_allele, alt_allele
                )

                ref_tensor = torch.tensor(
                    ref_seq, dtype=torch.float32
                ).unsqueeze(0).to(device)
                alt_tensor = torch.tensor(
                    alt_seq, dtype=torch.float32
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            ref_output = model(ref_tensor)
                            alt_output = model(alt_tensor)
                    else:
                        ref_output = model(ref_tensor)
                        alt_output = model(alt_tensor)

                if isinstance(ref_output, dict):
                    ref_pred = ref_output.get('atac_prediction', list(ref_output.values())[0])
                    alt_pred = alt_output.get('atac_prediction', list(alt_output.values())[0])
                else:
                    ref_pred = ref_output
                    alt_pred = alt_output

                ref_pred = ref_pred.squeeze()
                alt_pred = alt_pred.squeeze()

                seq_len = ref_pred.shape[0]
                crop_start = (seq_len - self.center_crop) // 2
                crop_end = crop_start + self.center_crop

                ref_sum = ref_pred[crop_start:crop_end].sum().item()
                alt_sum = alt_pred[crop_start:crop_end].sum().item()

                log_fc = np.log2((alt_sum + 1) / (ref_sum + 1))

                results.append({
                    'var.snp_id': row.get('var.snp_id', ''),
                    'var.chr': chrom,
                    'var.pos_hg38': pos,
                    'allele1': ref_allele,
                    'allele2': alt_allele,
                    'obs.meanLog2FC': row.get('obs.meanLog2FC', 0),
                    'pred_logfc': log_fc,
                    'ref_sum': ref_sum,
                    'alt_sum': alt_sum,
                })

            except Exception:
                continue

        return pd.DataFrame(results)
