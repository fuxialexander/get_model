import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import cooler
except ImportError:
    cooler = None

try:
    import hicstraw
except ImportError:
    hicstraw = None

import numpy as np
import pandas as pd
import seaborn as sns

try:
    from cooltools.lib.numutils import adaptive_coarsegrain
except ImportError:
    adaptive_coarsegrain = None

from matplotlib import pyplot as plt
from tqdm import tqdm


def _sample_contact_worker(args):
    """
    Worker function for parallel contact matrix sampling.

    Args:
        args: Tuple containing (hic_path, format, chrom, pos, matrix_size, resolution)

    Returns:
        numpy.ndarray: Contact matrix for the given position
    """
    hic_path, format, chrom, pos, matrix_size, resolution = args

    try:
        if format == "hicstraw":
            hic = hicstraw.HiCFile(hic_path)
            mzd = hic.getMatrixZoomData(
                chrom, chrom, "observed", "NONE", "BP", resolution
            )
            matrix = mzd.getRecordsAsMatrix(
                pos, pos + matrix_size * resolution, pos, pos + matrix_size * resolution
            )

        else:  # cooler
            hic = cooler.Cooler(f"{hic_path}::/resolutions/{resolution}")
            region = f"{chrom}:{pos}-{pos + matrix_size * resolution}"
            matrix = hic.matrix(balance=True).fetch(region)

        return np.nan_to_num(matrix)[0:matrix_size, 0:matrix_size]

    except Exception as e:
        return None


class HiCDataProcessor:
    """
    A class for processing HiC data from both hicstraw and cooler formats.
    Handles chromosome name formatting and resolution management automatically.
    """

    def __init__(self, hic_file: Union[str, Dict[str, object]]):
        """
        Initialize HiCDataProcessor with either a file path or HiC object.

        Args:
            hic_file: Path to .hic file or pre-loaded hicstraw/cooler object
        """
        if isinstance(hic_file, str):
            self.filename = hic_file
            if hic_file.endswith(".hic"):
                self.hic = hicstraw.HiCFile(hic_file)
                self.format = "hicstraw"
            elif hic_file.endswith(".cool") or hic_file.endswith(".mcool"):
                self.format = "cooler"
                try:  # if is single resolution cooler
                    self.hic = cooler.Cooler(hic_file)
                except:  # if is multi-resolution cooler
                    # get the first resolution
                    resolutions = cooler.fileops.list_coolers(hic_file)
                    self.hic = cooler.Cooler(f"{hic_file}::/{resolutions[0]}")
            else:
                raise ValueError(f"Unsupported file format: {hic_file}")
        else:
            self.filename = list(hic_file.keys())[0]
            self.hic = hic_file[self.filename]
            if hasattr(self.hic, "getMatrixZoomData"):
                self.format = "hicstraw"
            elif hasattr(self.hic, "matrix"):
                self.format = "cooler"
            else:
                raise ValueError(f"Unsupported file format: {hic_file}")

        if self.format == "hicstraw":
            self.chr_prefix = "chr" in self.hic.getChromosomes()[1].name
            self.resolutions = sorted(self.hic.getResolutions())
            self.assembly = self.hic.getGenomeID()
        elif self.format == "cooler":
            self.chr_prefix = "chr" in self.hic.chromnames[1]
            self.resolutions = sorted(
                [
                    int(r.split("/")[-1])
                    for r in cooler.fileops.list_coolers(self.hic.filename)
                ]
            )
            self.assembly = self.hic.info["genome-assembly"]

    def __repr__(self) -> str:
        """Return string representation with format and properties."""
        return (
            f"HiCDataProcessor(format='{self.format}', "
            f"chr_prefix={self.chr_prefix}, "
            f"resolutions={self.resolutions})"
        )

    def _format_chrom_name(self, chrom: str) -> str:
        """Ensure chromosome name matches storage format."""
        has_chr = chrom.startswith("chr")
        if has_chr == self.chr_prefix:
            return chrom
        return f"chr{chrom}" if self.chr_prefix else chrom.replace("chr", "")

    def generate_mean_contact_matrix(
        self,
        resolution: int = 5000,
        matrix_size: int = 1000,
        eps: float = 1e-3,
        max_iterations: int = 10000,
        save: bool = True,
        n_processes: int = os.cpu_count(),
    ) -> np.ndarray:
        """
        Generate and save mean contact matrix by randomly sampling across all chromosomes
        until convergence, using parallel processing.

        Args:
            resolution: Matrix resolution in bp (default: 5000)
            matrix_size: Size of contact matrix to analyze (default: 800)
            eps: Convergence threshold for mean change (default: 1e-3)
            max_iterations: Maximum number of iterations (default: 10000)
            save: Whether to save the result as .npy file (default: True)
            n_processes: Number of parallel processes to use (default: CPU count)

        Returns:
            numpy.ndarray: Mean contact matrix
        """
        matrix_size = matrix_size + 1  # somehow the first row and column are always 0
        # Get chromosome sizes
        if self.format == "hicstraw":
            chromosomes = self.hic.getChromosomes()
            chrom_sizes = {
                chr.name: chr.length
                for chr in chromosomes
                if chr.name.replace("chr", "").isdigit()
            }
        else:  # cooler
            chrom_sizes = {
                chrom: length
                for chrom, length in zip(self.hic.chromnames, self.hic.chromsizes)
                if chrom.replace("chr", "").isdigit()
            }

        # Generate random sampling points
        sampling_points = []
        for _ in range(max_iterations):
            chrom = np.random.choice(list(chrom_sizes.keys()))
            max_pos = chrom_sizes[chrom] - matrix_size * resolution
            if max_pos <= 0:
                continue
            pos = np.random.randint(0, max_pos)
            sampling_points.append(
                (self.filename, self.format, chrom, pos, matrix_size, resolution)
            )

        sum_count = np.zeros((matrix_size, matrix_size))
        count = 0
        prev_mean = None
        batch_size = 100  # Process in batches for convergence checking

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            pbar = tqdm(total=max_iterations)

            for batch_start in range(0, len(sampling_points), batch_size):
                batch_end = min(batch_start + batch_size, len(sampling_points))
                batch_points = sampling_points[batch_start:batch_end]

                # Process batch in parallel
                results = list(executor.map(_sample_contact_worker, batch_points))

                # Accumulate valid results
                for matrix in results:
                    if matrix is not None:
                        sum_count += matrix
                        count += 1
                        pbar.update(1)

                # Check convergence
                if count > 0:
                    current_mean = sum_count / count
                    if prev_mean is not None:
                        mean_change = np.mean(np.abs(current_mean - prev_mean))
                        if mean_change < eps:
                            print(
                                f"\nConverged after {count} iterations with mean change: {mean_change:.6f}"
                            )
                            break
                    prev_mean = current_mean.copy()

        pbar.close()
        mean_count = sum_count / count if count > 0 else sum_count
        mean_count = mean_count[1:, 1:]  # remove the first row and column
        if save:
            output_path = self.filename + f".mean_contact_{resolution}bp.npy"
            np.save(output_path, mean_count)
            print(f"Saved mean contact matrix to: {output_path}")

        return mean_count

    def get_mean_contact_map(self, resolution: int = 5000):
        """
        Get mean contact matrix for a given resolution.

        Args:
            resolution: Matrix resolution in bp

        Returns:
            numpy.ndarray: Mean contact matrix
        """
        mean_contacts_file = self.filename + f".mean_contact_{resolution}bp.npy"
        # load mean contacts if not in self
        if hasattr(self, "mean_contacts"):
            mean_contacts = self.mean_contacts
        elif os.path.exists(mean_contacts_file):
            mean_contacts = np.load(mean_contacts_file)
        else:
            mean_contacts = self.generate_mean_contact_matrix(
                resolution=resolution,
                matrix_size=1000,
                eps=1e-3,
                max_iterations=1000,
                save=True,
            )
        return mean_contacts

    def get_matrix(
        self,
        chrom: str,
        start: int,
        end: int,
        resolution: int = 5000,
        count_cutoff: int = 0,
        method: str = "oe",
        normalization: str = "SCALE",
        return_log: bool = True,
    ) -> np.ndarray:
        """
        Retrieve normalized HiC matrix for a given genomic region.

        Args:
            chrom: Chromosome name
            start: Start position (bp)
            end: End position (bp)
            resolution: Matrix resolution in bp
            count_cutoff: Minimum count threshold
            method: Matrix type ('observed', 'oe', etc.)
            normalization: Normalization method
            return_log: Whether to return log10-transformed matrix

        Returns:
            numpy.ndarray: Processed HiC matrix
        """
        # workaround to deal with 0/1-based indexing discrepancy
        end = end - 1
        chrom = self._format_chrom_name(chrom)

        if self.format == "hicstraw":
            mzd = self.hic.getMatrixZoomData(
                chrom, chrom, method, normalization, "BP", resolution
            )
            matrix = mzd.getRecordsAsMatrix(start, end, start, end)

            if count_cutoff > 0:
                mzd_raw = self.hic.getMatrixZoomData(
                    chrom, chrom, "observed", "NONE", "BP", resolution
                )
                raw_counts = mzd_raw.getRecordsAsMatrix(start, end, start, end)
                matrix[raw_counts <= count_cutoff] = 0

        elif self.format == "cooler":
            # change resolution if needed
            if self.hic.binsize != resolution:
                self.hic = cooler.Cooler(
                    f"{self.hic.filename}::/resolutions/{resolution}"
                )
            region = f"{chrom}:{start}-{end}"
            if normalization == "NONE":
                normalization = False
            else:
                normalization = True
            matrix = self.hic.matrix(balance=normalization).fetch(region)
            if count_cutoff > 0:
                raw_counts = self.hic.matrix(balance=False).fetch(region)
                matrix[raw_counts <= count_cutoff] = 0

        matrix = np.nan_to_num(matrix)
        return np.log10(matrix + 1) if return_log else matrix

    def get_coarse_grain_matrix(
        self,
        chrom: str,
        start: int,
        end: int,
        resolution: int = 5000,
        count_cutoff: int = 5,
        method: str = "oe",
        normalization: str = "KR",
        return_log: bool = True,
    ) -> np.ndarray:
        """
        Get coarse-grained HiC unnormalized matrix using adaptive coarsegraining.

        Args:
            chrom: Chromosome name
            start: Start position (bp)
            end: End position (bp)
            resolution: Matrix resolution in bp
            count_cutoff: Minimum count threshold
            method: Matrix type, 'oe' or 'observed'
            normalization: Normalization method, 'NONE', 'SCALE', 'VC', 'VC_SQRT'
            return_log: Whether to return log10-transformed matrix

        Returns:
            numpy.ndarray: Coarse-grained matrix
            numpy.ndarray: Raw counts matrix
        """
        # get mean contacts
        mean_contacts = self.get_mean_contact_map(resolution=resolution)

        # Get raw counts
        raw_counts = self.get_matrix(
            chrom,
            start,
            end,
            resolution,
            method="observed",
            normalization="NONE",
            return_log=False,
        )

        # Get normalized matrix
        norm_matrix = self.get_matrix(
            chrom,
            start,
            end,
            resolution,
            method="observed",
            normalization=normalization,
            return_log=False,
        )

        # trim mean contacts to matrix size
        mean_contacts = mean_contacts[
            0 : norm_matrix.shape[0], 0 : norm_matrix.shape[1]
        ]

        # Perform adaptive coarsegraining
        # don't report RuntimeWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            coarse = adaptive_coarsegrain(norm_matrix, raw_counts, cutoff=count_cutoff)
        coarse = np.nan_to_num(coarse, posinf=0, neginf=0)
        coarse = coarse[0 : norm_matrix.shape[0], 0 : norm_matrix.shape[1]]

        # Calculate final matrix
        if method == "oe":
            result = (coarse + 1) / (mean_contacts + 1)
        else:
            result = coarse + 1

        if return_log:
            result = np.log10(result)

        # Mask zero-sum rows/columns and adjacent rows/columns
        zero_mask = np.where(raw_counts.sum(0) <= count_cutoff)[0]
        zero_mask = np.concatenate([zero_mask, zero_mask + 1, zero_mask - 1])
        zero_mask = np.unique(
            zero_mask[(zero_mask >= 0) & (zero_mask < result.shape[0])]
        )
        result[zero_mask, :] = 0
        result[:, zero_mask] = 0
        return result, raw_counts

    def get_submatrix_from_regions(
        self,
        regions_df: pd.DataFrame,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        coarse_grain: bool = False,
        resolution: int = 5000,
        count_cutoff: int = 5,
        method: str = "oe",
        normalization: str = "SCALE",
        return_log: bool = True,
        max_region_size: int = 4000000,
        mask_zero: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Extract HiC submatrix for genomic regions defined in a DataFrame.

        Args:
            regions_df: DataFrame with Chromosome, Start, End columns
            start_idx: Start index in DataFrame
            end_idx: End index in DataFrame
            coarse_grain: Whether to use coarse-grained matrix. If True, will use adaptive coarsegraining, and method, normalization, and count_cutoff will be ignored.
            resolution: Matrix resolution in bp
            method: Matrix type
            normalization: Normalization method
            count_cutoff: Minimum count threshold
            return_log: Whether to return log10-transformed matrix
            max_region_size: Maximum region size in bp
            mask_zero: Whether to mask zero-sum rows/columns based on raw counts
        Returns:
            numpy.ndarray or None: Submatrix if successful, None if invalid region
        """
        regions = (
            regions_df.iloc[start_idx:end_idx] if start_idx is not None else regions_df
        )

        # Validate region
        if regions.Chromosome.nunique() > 1:
            return None

        chrom = self._format_chrom_name(regions.iloc[0].Chromosome)
        start = regions.iloc[0].Start // resolution
        end = regions.iloc[-1].End // resolution + 1

        if (end - start) * resolution > max_region_size:
            return None

        # Get indices for subsetting
        indices = np.array(
            [row.Start // resolution - start for _, row in regions.iterrows()]
        )

        # Get full matrix
        if coarse_grain:
            full_matrix, raw_counts = self.get_coarse_grain_matrix(
                chrom=chrom,
                start=start * resolution,
                end=end * resolution,
                resolution=resolution,
                count_cutoff=count_cutoff,
                method=method,
                normalization=normalization,
                return_log=return_log,
            )
            print("raw_counts.shape", raw_counts.shape)
            print("full_matrix.shape", full_matrix.shape)
            if mask_zero:
                full_matrix[raw_counts==0] = 0
        else:
            full_matrix = self.get_matrix(
                chrom=chrom,
                start=start * resolution,
                end=end * resolution,
                resolution=resolution,
                count_cutoff=count_cutoff,
                method=method,
                normalization=normalization,
                return_log=return_log,
            )
        return full_matrix[indices, :][:, indices]

    def plot_matrix(
        self,
        chrom: str,
        start: int,
        end: int,
        coarse_grain: bool = False,
        resolution: int = 5000,
        count_cutoff: int = 5,
        method: str = "oe",
        normalization: str = "SCALE",
        ax: Optional[plt.Axes] = None,
        **kargs,
    ):
        """
        Plot HiC matrix for a given genomic region.

        Args:
            chrom: Chromosome name
            start: Start position (bp)
            end: End position (bp)
            coarse_grain: Whether to use coarse-grained matrix. If True, will use adaptive coarsegraining, and method, normalization, and count_cutoff will be ignored.
            resolution: Matrix resolution in bp
            count_cutoff: Minimum count threshold
            method: Matrix type, 'oe' or 'observed'
            normalization: Normalization method, 'NONE', 'SCALE', 'KR', 'VC', 'VC_SQRT'
            ax: Matplotlib axis object
            kargs: Additional keyword arguments for sns.heatmap
        """
        if coarse_grain:
            matrix, raw_counts = self.get_coarse_grain_matrix(
                chrom,
                start,
                end,
                resolution=resolution,
                count_cutoff=count_cutoff,
                method=method,
                normalization=normalization,
                return_log=True,
            )
        else:
            matrix = self.get_matrix(
                chrom,
                start,
                end,
                resolution=resolution,
                method=method,
                normalization=normalization,
                count_cutoff=count_cutoff,
                return_log=True,
            )
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(matrix, ax=ax, cbar=False, **kargs)
        return ax

    def plot_submatrix(
        self,
        regions_df: pd.DataFrame,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        **kargs,
    ):
        """
        Plot HiC submatrix for genomic regions defined in a DataFrame.
        """
        submatrix = self.get_submatrix_from_regions(
            regions_df,
            start_idx=start_idx,
            end_idx=end_idx,
            return_original=False,
            **kargs,
        )
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(submatrix, ax=ax, cbar=False, **kargs)
        return ax


def get_hic_from_idx(
    hic: Union[str, Dict[str, object], HiCDataProcessor],
    csv: pd.DataFrame,
    start: Optional[int] = None,
    end: Optional[int] = None,
    resolution: int = 5000,
    method: str = "oe",
    normalization: str = "SCALE",
    count_cutoff: int = 5,
    coarse_grain: bool = True,
):
    """
    Get HiC submatrix from a DataFrame of regions. For backward compatibility with old code.
    """
    if not isinstance(hic, HiCDataProcessor) and isinstance(hic, dict):
        hic = HiCDataProcessor(hic)
    elif not isinstance(hic, HiCDataProcessor):
        raise ValueError(f"Unsupported hic object: {hic}")
    return hic.get_submatrix_from_regions(
        csv,
        start_idx=start,
        end_idx=end,
        coarse_grain=coarse_grain,
        resolution=resolution,
        count_cutoff=count_cutoff,
        method=method,
        normalization=normalization,
        return_log=True,
        max_region_size=4000000,
        mask_zero=True,
    )
