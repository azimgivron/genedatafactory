from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
import torch 

from genedatafactory.synthetic.models import (FixedGCN,
                                              FixedMLP)
from genedatafactory.synthetic.synthetic_config import SyntheticConfig


def sample_gene_latents(
    config: SyntheticConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample latent gene variables H along with their mixture parameters.

    Args:
        config (SyntheticConfig): Mixture sizes and noise scales.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (H, pi,
        gene_component_means, z_genes) where H has shape (n_genes,
        L_gene_latent), pi are mixture weights, gene_component_means are
        component means, and z_genes are mixture assignments.
    """
    pi = np.random.dirichlet([config.dirichlet_conc_genes] * config.D_gene_components)
    gene_component_means = np.random.randn(
        config.D_gene_components, config.L_gene_latent
    ).astype(np.float32)
    z_genes = np.random.choice(config.D_gene_components, size=config.n_genes, p=pi)

    H = np.zeros((config.n_genes, config.L_gene_latent), dtype=np.float32)
    for i, component in enumerate(z_genes):
        H[i] = gene_component_means[component] + config.sigma_H * np.random.randn(
            config.L_gene_latent
        )
    return H, pi, gene_component_means, z_genes


def sample_gene_features_and_graph(
    config: SyntheticConfig, H: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate gene features X from latent H and build a random geometric graph.

    Args:
        config (SyntheticConfig): Feature sizes and graph scale.
        H (ndarray): Latent gene matrix of shape (n_genes, L_gene_latent).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (X, A_X, A_mat) with numpy
        features, readout matrix, and the symmetric adjacency matrix.
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

    return X.astype(np.float32), A_X, A_mat


def compute_gene_factors(
    config: SyntheticConfig, H: np.ndarray, edge_index: np.ndarray, device: str
) -> np.ndarray:
    """Apply a linear GCN-style projection to produce noisy gene factors U.

    Args:
        config (SyntheticConfig): Architecture and noise settings.
        H (ndarray): Gene features.
        edge_index (ndarray): Edge indices for the gene graph.

    Returns:
        np.ndarray: Shape (n_genes, k_latent) containing gene factors.
    """
    gnn = FixedGCN(
        in_dim=config.L_gene_latent,
        hidden_dims=config.gnn_hidden_dims,
        out_dim=config.k_latent,
    ).to(device)
    with torch.no_grad():
        mu_U = gnn(torch.from_numpy(H).to(device), torch.from_numpy(edge_index).to(device))
    return (mu_U + config.sigma_U * torch.randn_like(mu_U)).numpy()


def sample_disease_features(
    config: SyntheticConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample disease features D from a Gaussian mixture and return mixture stats.

    Args:
        config (SyntheticConfig): Holds mixture sizes and feature dimensions.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (D, rho,
        disease_component_means, z_diseases) where D has shape
        (n_diseases, L_disease_latent).
    """
    rho = np.random.dirichlet(
        [config.dirichlet_conc_diseases] * config.C_disease_components
    )
    disease_component_means = np.random.randn(
        config.C_disease_components, config.L_disease_latent
    ).astype(np.float32)
    z_diseases = np.random.choice(
        config.C_disease_components, size=config.n_diseases, p=rho
    )

    D = np.zeros((config.n_diseases, config.L_disease_latent), dtype=np.float32)
    for j, component in enumerate(z_diseases):
        D[j] = disease_component_means[component] + config.sigma_D * np.random.randn(
            config.L_disease_latent
        )
    return D, rho, disease_component_means, z_diseases


def compute_disease_factors(
    config: SyntheticConfig, D: np.ndarray, device: str
) -> np.ndarray:
    """Apply a linear projection to produce noisy disease factors W.

    Args:
        config (SyntheticConfig): Architecture and noise settings.
        D (np.ndarray): Disease latents.
        device (str): Device on which to run the network.

    Returns:
        np.ndarray: Shape (n_diseases, k_latent) containing disease factors.
    """
    disease_mlp = FixedMLP(
        in_dim=config.L_disease_latent,
        hidden_dims=config.disease_mlp_hidden_dims,
        out_dim=config.k_latent,
    ).to(device)
    with torch.no_grad():
        mu_W = disease_mlp(torch.from_numpy(D).to(device))
    return (mu_W + config.sigma_W * torch.randn_like(mu_W)).numpy()


def sample_interactions(
    U: np.ndarray, W: np.ndarray, bias: float, sigma_z: float
) -> np.ndarray:
    """Sample binary interactions R with a probit link over U and W.

    Args:
        U (ndarray): Gene factor matrix of shape (n_genes, k_latent).
        W (ndarray): Disease factor matrix of shape (n_diseases, k_latent).
        bias (float): Bias term inside the probit link.
        sigma_z (float): Noise scale in the probit link.

    Returns:
        np.ndarray: Shape (n_genes, n_diseases) with binary interactions.
    """
    scores = U @ W.T
    probs = norm.cdf((scores + bias) / sigma_z)
    return np.random.binomial(1, probs)
