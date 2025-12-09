import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from genedatafactory.synthetic.sample_compute import (
    compute_disease_factors, compute_gene_factors, sample_disease_features,
    sample_gene_features_and_graph, sample_gene_latents, sample_interactions)
from genedatafactory.synthetic.synthetic_config import SyntheticConfig


def generate_synthetic_dataset(
    config: SyntheticConfig,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Generate a synthetic gene–disease dataset with hierarchical structure.

    The pipeline seeds RNGs, samples gene latents and features, builds a random
    geometric graph, pushes features through frozen networks to obtain factors,
    and samples a binary interaction matrix with a probit link.

    Args:
        config (SyntheticConfig): Controls sizes, noise scales, and architectures.
        seed (int): Random seed applied to numpy and torch.
        device (str): Torch device for generated tensors.

    Returns:
        dict[str, np.ndarray]: Mapping containing numpy arrays for R, U, W,
        X, Y, edge_index, H_latent_genes, z_genes, gene_mixture_weights,
        gene_component_means, z_diseases, disease_mixture_weights,
        disease_component_means, and A_X.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    H, pi, gene_component_means, z_genes = sample_gene_latents(config)
    X_t, A_X, edge_index = sample_gene_features_and_graph(config, H, device)
    U_t = compute_gene_factors(config, X_t, edge_index, device)

    Y, rho, disease_component_means, z_diseases = sample_disease_features(config)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    W_t = compute_disease_factors(config, Y_t, device)

    R_t = sample_interactions(U_t, W_t, config.bias, config.sigma_z)

    out = {
        "R": np.array(R_t, copy=False),
        "U": U_t.detach().cpu().numpy(),
        "W": W_t.detach().cpu().numpy(),
        "X": X_t.detach().cpu().numpy(),
        "Y": Y_t.detach().cpu().numpy(),
        "edge_index": edge_index.detach().cpu().numpy(),
        "H_latent_genes": np.array(H, copy=False),
        "z_genes": np.array(z_genes, copy=False),
        "gene_mixture_weights": np.array(pi, copy=False),
        "gene_component_means": np.array(gene_component_means, copy=False),
        "z_diseases": np.array(z_diseases, copy=False),
        "disease_mixture_weights": np.array(rho, copy=False),
        "disease_component_means": np.array(disease_component_means, copy=False),
        "A_X": np.array(A_X, copy=False),
    }
    return out


def dense_to_coo(
    df_array: np.ndarray, row_name: str, col_name: str, value_name: str = "Value"
) -> pd.DataFrame:
    """
    Convert a dense 2D numpy array into a COO-like DataFrame:
    (row_index, col_index, value), only for non-zero entries.
    """
    if df_array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {df_array.shape}")

    rows, cols = np.nonzero(df_array)  # indices of non-zero entries
    values = df_array[rows, cols]

    return pd.DataFrame(
        {
            row_name: rows,
            col_name: cols,
            value_name: values,
        }
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input and output directories.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=("🧬 Generated Synthetic Gene-Disease Data."),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Folder to write processed CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = SyntheticConfig()

    data = generate_synthetic_dataset(cfg, seed=42, device="cpu")

    association = data["R"]  # shape: (n_genes, n_diseases)
    gene = data["X"]  # shape: (n_genes, n_gene_features)
    disease = data["Y"]  # shape: (n_diseases, n_disease_features)
    ppi = data["A_X"]  # assuming dense adjacency: (n_genes, n_genes)

    print(
        "association shape:",
        association.shape,
        association.sum() / association.size * 100,
    )
    print("gene shape:", gene.shape)
    print("disease shape:", disease.shape)
    print("ppi shape:", ppi.shape)

    # Convert all dense matrices to COO-like triplets
    assoc_df = dense_to_coo(
        association, row_name="Gene ID", col_name="Disease ID", value_name="Value"
    )
    gene_df = dense_to_coo(
        gene, row_name="Gene ID", col_name="Feature ID", value_name="Value"
    )
    disease_df = dense_to_coo(
        disease, row_name="Disease ID", col_name="Feature ID", value_name="Value"
    )
    ppi_df = dense_to_coo(ppi, row_name="Gene_i", col_name="Gene_j", value_name="Value")

    # Save to CSV (without pandas index)
    assoc_df.to_csv(args.output / "gene_disease.csv", index=False)
    gene_df.to_csv(args.output / "gene_si.csv", index=False)
    disease_df.to_csv(args.output / "disease_si.csv", index=False)
    ppi_df.to_csv(args.output / "string.csv", index=False)
