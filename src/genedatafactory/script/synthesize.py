import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from genedatafactory.synthetic.sample_compute import (
    compute_disease_factors, compute_gene_factors, sample_disease_features,
    sample_gene_features_and_graph, sample_gene_latents, sample_interactions)
from genedatafactory.synthetic.synthetic_config import SyntheticConfig


def generate_synthetic_dataset(
    config: SyntheticConfig,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Generate a synthetic gene–disease dataset with hierarchical structure.

    The pipeline seeds RNGs, samples gene latents and features, builds a random
    geometric graph, pushes features through frozen networks to obtain factors,
    and samples a binary interaction matrix with a probit link.

    Args:
        config (SyntheticConfig): Controls sizes, noise scales, and architectures.
        seed (int): Random seed applied to numpy RNG.

    Returns:
        dict[str, np.ndarray]: Mapping containing numpy arrays for R, U, W,
        X, Y, edge_index, H_latent_genes, D_latent_disease, z_genes, gene_mixture_weights,
        gene_component_means, z_diseases, disease_mixture_weights,
        disease_component_means, and A_X and A_Y.
    """
    np.random.seed(seed)

    H, pi, gene_component_means, z_genes = sample_gene_latents(config)
    X, A_X, A_mat = sample_gene_features_and_graph(config, H)
    row_idx, col_idx = np.nonzero(A_mat)
    edge_index = np.vstack([row_idx, col_idx]).astype(np.int64)
    U = compute_gene_factors(config, H, A_mat)

    D, rho, disease_component_means, z_diseases = sample_disease_features(config)
    A_Y = np.random.randn(config.d_Y, config.L_disease_latent).astype(np.float32)
    Y = (A_Y @ D.T).T
    Y += config.sigma_Y * np.random.randn(config.n_diseases, config.d_Y).astype(np.float32)
    W = compute_disease_factors(config, D, U)

    R = sample_interactions(U, W, config.bias, config.sigma_z)

    out = {
        "R": np.array(R, copy=False),
        "U": np.array(U, copy=False),
        "W": np.array(W, copy=False),
        "X": np.array(X, copy=False),
        "Y": np.array(Y, copy=False),
        "edge_index": np.array(edge_index, copy=False),
        "H_latent_genes": np.array(H, copy=False),
        "D_latent_disease": np.array(D, copy=False),
        "z_genes": np.array(z_genes, copy=False),
        "gene_mixture_weights": np.array(pi, copy=False),
        "gene_component_means": np.array(gene_component_means, copy=False),
        "z_diseases": np.array(z_diseases, copy=False),
        "disease_mixture_weights": np.array(rho, copy=False),
        "disease_component_means": np.array(disease_component_means, copy=False),
        "A_X": np.array(A_X, copy=False),
        "A_Y": np.array(A_Y, copy=False),
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
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    cfg = SyntheticConfig()

    data = generate_synthetic_dataset(cfg, seed=42)

    association = data["R"]  # shape: (n_genes, n_diseases)
    gene = data["X"]  # shape: (n_genes, n_gene_features)
    disease = data["Y"]  # shape: (n_diseases, n_disease_features)
    ppi = data["edge_index"]  # COO edge list of connected genes

    print(
        "association shape:",
        association.shape,
        f"{association.sum() / association.size * 100:.3f}%",
    )
    print("gene shape:", gene.shape)
    print("disease shape:", disease.shape)
    print("ppi shape:", ppi.shape)

    # Convert all dense matrices to COO-like triplets
    assoc_df = dense_to_coo(
        association, row_name="Gene ID", col_name="Disease ID", value_name="Value"
    ).drop(columns=["Value"])
    gene_df = dense_to_coo(
        gene, row_name="Gene ID", col_name="Feature ID", value_name="Value"
    )
    disease_df = dense_to_coo(
        disease, row_name="Disease ID", col_name="Feature ID", value_name="Value"
    )
    ppi_df = pd.DataFrame(ppi.T, columns=["Gene_i", "Gene_j"])

    # Save to CSV (without pandas index)
    assoc_df.to_csv(output / "gene_disease.csv", index=False)
    gene_df.to_csv(output / "gene_si.csv", index=False)
    disease_df.to_csv(output / "disease_si.csv", index=False)
    ppi_df.to_csv(output / "string.csv", index=False)

    meta = {
        "nb_genes": association.shape[0],
        "nb_diseases": association.shape[1]
    }
    with open(output/"meta.yaml", "w") as stream:
        yaml.dump(meta, stream)
