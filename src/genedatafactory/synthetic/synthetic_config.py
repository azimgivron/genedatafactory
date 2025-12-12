from dataclasses import dataclass


@dataclass
class SyntheticConfig:
    """Hyperparameters and sizes that control the synthetic data generator.

    Attributes:
        n_genes (int): Number of genes (nodes) to sample.
        n_diseases (int): Number of diseases to sample.
        k_latent (int): Dimension of latent gene and disease factors U and W.
        d_X (int): Gene feature dimension.
        d_Y (int): Disease feature dimension.
        L_gene_latent (int): Dimension of latent H_i vectors.
        D_gene_components (int): Number of Gaussian mixture components for genes.
        C_disease_components (int): Number of Gaussian mixture components for diseases.
        gnn_hidden_dims (tuple): Linear layer sizes for the fixed 3-hop graph projection.
        disease_mlp_hidden_dims (tuple): Linear layer sizes for the fixed disease projection.
        sigma_H (float): Stddev for sampling latent gene components.
        sigma_X (float): Stddev for gene feature noise.
        sigma_Y (float): Stddev for disease feature noise.
        sigma_U (float): Stddev for gene factor noise.
        sigma_W (float): Stddev for disease factor noise.
        eta (float): Length scale for the random geometric graph.
        dirichlet_conc_genes (float): Dirichlet concentration for gene mixture weights.
        dirichlet_conc_diseases (float): Dirichlet concentration for disease mixture weights.
        sigma_z (float): Probit noise scale for interaction sampling.
        bias (float): Bias term inside the probit link.
    """

    # Sizes
    n_genes: int = 5000
    n_diseases: int = 100
    k_latent: int = 40  # dimension of U_i and W_j
    d_X: int = 256  # gene feature dimension
    d_Y: int = 256  # disease feature dimension
    L_gene_latent: int = 64  # dimension of H_i
    L_disease_latent: int = 64  # dimension of D_j
    D_gene_components: int = 100  # # gene mixture components
    C_disease_components: int = 20  # # disease mixture components

    # GNN and MLP architecture
    gnn_hidden_dims: tuple = (128, 64, 32)  # layers for f0(X, G)
    disease_mlp_hidden_dims: tuple = (64, 32)  # layers for g0(Y)

    # Noise scales / variances
    sigma_H: float = 0.05
    sigma_D: float = 0.05
    sigma_X: float = 0.05
    sigma_Y: float = 0.05
    sigma_U: float = 0.05
    sigma_W: float = 0.05

    # Random geometric graph scale
    eta: float = .5

    # Mixture Dirichlet concentrations
    dirichlet_conc_genes: float = 1.0
    dirichlet_conc_diseases: float = 1.0

    # Binary causal link
    sigma_z: float = 1.0
    bias: float = -2.4e0
