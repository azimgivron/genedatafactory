from typing import Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

from genedatafactory.synthetic.models import FixedGCN, FixedMLP
from genedatafactory.synthetic.synthetic_config import SyntheticConfig


def sample_gene_latents(
    config: SyntheticConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample latent gene variables H along with their mixture parameters.

    Args:
        config (SyntheticConfig): Holds mixture sizes and noise scales.

    Returns:
        tuple: (H, pi, gene_component_means, z_genes) where:
            H (ndarray): Shape (n_genes, L_gene_latent) with latent embeddings.
            pi (ndarray): Mixture weights for genes.
            gene_component_means (ndarray): Component means for H.
            z_genes (ndarray): Mixture assignments for each gene.
    """
    pi = np.random.dirichlet([config.dirichlet_conc_genes] * config.D_gene_components)
    gene_component_means = np.random.randn(
        config.D_gene_components, config.L_gene_latent
    )
    z_genes = np.random.choice(config.D_gene_components, size=config.n_genes, p=pi)

    H = np.zeros((config.n_genes, config.L_gene_latent), dtype=np.float32)
    for i, component in enumerate(z_genes):
        H[i] = gene_component_means[component] + config.sigma_H * np.random.randn(
            config.L_gene_latent
        )
    return H, pi, gene_component_means, z_genes


def sample_gene_features_and_graph(
    config: SyntheticConfig, H: np.ndarray, device: str
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Generate gene features X from latent H and build a random geometric graph.

    Args:
        config (SyntheticConfig): Feature sizes and graph scale.
        H (ndarray): Latent gene matrix of shape (n_genes, L_gene_latent).
        device (str): Torch device for returned tensors.

    Returns:
        tuple: (X_t, A_X, edge_index) with torch features, readout matrix,
        and edge indices for the gene graph.
    """
    A_X = np.random.randn(config.d_X, config.L_gene_latent).astype(np.float32)
    X = (A_X @ H.T).T
    X += config.sigma_X * np.random.randn(config.n_genes, config.d_X).astype(np.float32)

    pairwise_sqdist = squareform(pdist(H, metric="sqeuclidean"))
    P_edges = np.exp(-pairwise_sqdist / (2.0 * config.eta**2))
    np.fill_diagonal(P_edges, 0.0)

    upper = np.random.rand(config.n_genes, config.n_genes)
    A_mat = (upper < P_edges).astype(np.int8)
    A_mat = np.triu(A_mat, 1)
    A_mat = A_mat + A_mat.T

    row_idx, col_idx = np.nonzero(A_mat)
    edge_index = torch.tensor(
        np.vstack([row_idx, col_idx]),
        dtype=torch.long,
        device=device,
    )
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    return X_t, A_X, edge_index


def compute_gene_factors(
    config: SyntheticConfig, X_t: torch.Tensor, edge_index, device
) -> torch.Tensor:
    """Apply the frozen GCN to produce noisy gene factors U.

    Args:
        config (SyntheticConfig): Architecture and noise settings.
        X_t (Tensor): Gene features on the target device.
        edge_index (Tensor): Edge indices for the gene graph.
        device (str): Torch device for the model.

    Returns:
        Tensor: Shape (n_genes, k_latent) containing gene factors.
    """
    gnn = FixedGCN(
        in_dim=config.d_X,
        hidden_dims=config.gnn_hidden_dims,
        out_dim=config.k_latent,
    ).to(device)
    with torch.no_grad():
        mu_U = gnn(X_t, edge_index)
    return mu_U + config.sigma_U * torch.randn_like(mu_U)


def sample_disease_features(
    config: SyntheticConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample disease features Y from a Gaussian mixture and return mixture stats.

    Args:
        config (SyntheticConfig): Holds mixture sizes and feature dimensions.

    Returns:
        tuple: (Y, rho, disease_component_means, z_diseases) where Y has
        shape (n_diseases, d_Y).
    """
    rho = np.random.dirichlet(
        [config.dirichlet_conc_diseases] * config.C_disease_components
    )
    disease_component_means = np.random.randn(
        config.C_disease_components, config.d_Y
    ).astype(np.float32)
    z_diseases = np.random.choice(
        config.C_disease_components, size=config.n_diseases, p=rho
    )

    Y = np.zeros((config.n_diseases, config.d_Y), dtype=np.float32)
    for j, component in enumerate(z_diseases):
        Y[j] = disease_component_means[component] + config.sigma_Y * np.random.randn(
            config.d_Y
        )
    return Y, rho, disease_component_means, z_diseases


def compute_disease_factors(
    config: SyntheticConfig, Y_t: torch.Tensor, device
) -> torch.Tensor:
    """Apply the frozen MLP to produce noisy disease factors W.

    Args:
        config (SyntheticConfig): Architecture and noise settings.
        Y_t (Tensor): Disease features on the target device.
        device (str): Torch device for the model.

    Returns:
        Tensor: Shape (n_diseases, k_latent) containing disease factors.
    """
    disease_mlp = FixedMLP(
        in_dim=config.d_Y,
        hidden_dims=config.disease_mlp_hidden_dims,
        out_dim=config.k_latent,
    ).to(device)
    with torch.no_grad():
        mu_W = disease_mlp(Y_t)
    return mu_W + config.sigma_W * torch.randn_like(mu_W)


def sample_interactions(
    U_t: torch.Tensor, W_t: torch.Tensor, bias: float, sigma_z: float
) -> np.ndarray:
    """Sample binary interactions R with a probit link over U and W.

    Args:
        U_t (Tensor): Gene factor matrix of shape (n_genes, k_latent).
        W_t (Tensor): Disease factor matrix of shape (n_diseases, k_latent).
        bias (float): Bias term inside the probit link.
        sigma_z (float): Noise scale in the probit link.

    Returns:
        ndarray: Shape (n_genes, n_diseases) with binary interactions.
    """
    scores = U_t @ W_t.t()
    probs = norm.cdf((scores + bias) / sigma_z)
    rng = np.random.default_rng()
    return rng.binomial(1, probs)
