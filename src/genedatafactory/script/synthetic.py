from typing import Dict

import numpy as np
import torch

from genedatafactory.synthetic.sample_compute import (
    compute_disease_factors,
    compute_gene_factors,
    sample_disease_features,
    sample_gene_features_and_graph,
    sample_gene_latents,
    sample_interactions,
)
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


if __name__ == "__main__":
    cfg = SyntheticConfig()

    data = generate_synthetic_dataset(cfg, seed=42, device="cpu")

    R = data["R"]
    U = data["U"]
    W = data["W"]
    X = data["X"]
    Y = data["Y"]

    print("R shape:", R.shape, R.sum() / R.size * 100)
    print("U shape:", U.shape)
    print("W shape:", W.shape)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
