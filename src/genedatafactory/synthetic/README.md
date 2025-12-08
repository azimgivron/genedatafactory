# Data Generation
## 0. Notation and goal
- $n$: number of genes (row entities).
- $m$: number of diseases (column entities).
- $k$: latent dimension of gene and disease factors.
- $d_X$: dimension of gene features $X_i \in \mathbb{R}^{d_X}$.
- $d_Y$: dimension of disease features $Y_j \in \mathbb{R}^{d_Y}$.
- $L$: dimension of latent embedding $H_i \in \mathbb{R}^L$.
- $D$: number of **gene mixture components** (modes).
- $C$: number of **disease mixture components** (modes).

We model a real-valued gene–disease interaction matrix:  
$$
R \in \mathbb{R}^{n \times m},\quad R_{ij} \text{ = interaction strength between gene } i \text{ and disease } j
$$
## 1. Overall matrix factor model

We introduce latent factors:
- For each gene $i$: $U_i \in \mathbb{R}^k$.
- For each disease $j$: $W_j \in \mathbb{R}^k$.

We also introduce a latent continuous “causal strength” variable $Z_{ij}$:
$$Z_{ij} \mid U_i, W_j \sim \mathcal{N}\big(U_i^\top W_j + b,\; \sigma_Z^2\big),$$
Then define the **binary link** via a threshold:
$$
R_{ij} =
\begin{cases}
1 & \text{if } Z_{ij} > 0,\\
0 & \text{otherwise}
\end{cases}
$$
Equivalently, marginalizing out $Z_{ij}$:
$$
\mathbb{P}(R_{ij}=1 \mid U_i, W_j)
= \Phi\left( \dfrac{U_i^\top W_j + b}{\sigma_Z} \right)
$$
where $\Phi(\cdot)$ is the standard normal CDF. So:
$$
R_{ij} \mid U_i, W_j \sim \text{Bernoulli}\left( \Phi\left( \dfrac{U_i^\top W_j + b}{\sigma_Z} \right) \right)
$$
This is a **probit-type binary factor model** where the continuous interaction strength is only observed through a yes/no causal link.

## 2. Gene side (rows): latent embedding, features, and graph

We generate both gene features $X$ and a gene graph $G$ from a shared latent embedding $H$. This makes $G$ informative about the same underlying biology as $X$, but in a complementary way.

For each gene $i \in {1,\dots,n}$:
- **Mixture assignment (discrete):**  
    $$
    z_i^X \sim \text{Categorical}(\pi_1,\dots,\pi_D),\quad \sum_{d=1}^D \pi_d = 1
    $$

Given $z_i^X$, we draw a latent embedding $H_i \in \mathbb{R}^L$:
	$$
	H_i \mid z_i^X = d \sim \mathcal{N}(\tilde{\mu}_d,\ \tilde{\Sigma}_d),\quad d = 1,\dots,D
	$$
- Marginally over $z_i^X$, $H_i$ is drawn from a **Gaussian mixture**:  
	$$
	H_i \sim \sum_{d=1}^D \pi_d\, \mathcal{N}(\tilde{\mu}_d,\tilde{\Sigma}_d)
	$$
Observed gene features $X_i \in \mathbb{R}^{d_X}$ are a noisy linear readout of $H_i$:
	$$
	X_i \mid H_i \sim \mathcal{N}(A H_i,\ \sigma_X^2 I_{d_X}),\quad A \in \mathbb{R}^{d_X \times L}
	$$
- Because $H_i$ is a Gaussian mixture, the marginal distribution of $X_i$ is a **Gaussian mixture**.

We build an undirected graph $G = (V,E)$ on the genes, with $V = {1,\dots,n}$, adjacency matrix $A = (A_{i\ell})$.

For each unordered pair $(i,\ell)$, $i<\ell$:
$$
A_{i\ell} \mid H_i, H_\ell \sim \text{Bernoulli}(p_{i\ell}),\quad  
p_{i\ell} = \exp\left(-\frac{\|H_i - H_\ell\|^2}{2\eta^2}\right) 
$$
for some scale parameter $\eta > 0$.
- This defines a **random geometric graph** in latent space.

## 3. Non-linear prior for $U$ via fixed graph-based $f_0(X,G)$

We want $U_i$ to be a **non-linear function** of both $X$ and the gene graph $G$, while keeping the conditional distribution Gaussian. Let $\tilde{A}$ be a normalized adjacency matrix of $G$ (e.g. symmetric normalization). Define a fixed GNN-like mapping with **non-learned, fixed parameters**:
$$
\begin{align}
H^{(1)} &= \sigma(\tilde{A} X W_1 + b_1)\\
H^{(2)} &= \sigma(\tilde{A} H^{(1)} W_2 + b_2)\\
\vdots \\
H^{(L_G)} &= \sigma(\tilde{A} H^{(L_G-1)} W_{L_G} + b_{L_G})\\
f_0(X,G) &= H^{(L_G)} W_{\text{out}} + b_{\text{out}} 
\end{align}
$$
Where:
- $\sigma(\cdot)$ is a pointwise nonlinearity (e.g. ReLU, $\tanh$).
- $W_\ell, b_\ell, W_{\text{out}}, b_{\text{out}}$ are **fixed matrices/vectors**, sampled once (e.g. from Gaussian initializations) and never trained.

Thus, for each gene $i$:  
$$
\mu^{(U)}_i = [f_0(X,G)]_i \in \mathbb{R}^k
$$
We define:  
$$U_i \mid X,G \sim \mathcal{N}\big(\mu^{(U)}_i,\ \Sigma^{(U)}\big),\quad  
\Sigma^{(U)} = \sigma_U^2 I_k.  $$
- The mean $\mu^{(U)}_i$ is a **non-linear function** of ($X,G$) due to the GNN-like forward pass.

## 4. Disease side (columns): mixture features and non-linear $g_0(Y)$

We now specify the disease side. There is **no disease graph**. We only use a Gaussian mixture for disease features $Y$ and a non-linear function $g_0$ to generate $W_j$.

For each disease $j \in {1,\dots,m}$:
- **Mixture assignment:**  
	$$
	z_j^Y \sim \text{Categorical}(\rho_1,\dots,\rho_C),\quad \sum_{c=1}^C \rho_c = 1
	$$
Distribution: **Categorical** over $C$ modes. Given $z_j^Y = c$, draw features $Y_j \in \mathbb{R}^{d_Y}$:
	$$
	Y_j \mid z_j^Y = c \sim \mathcal{N}(\nu_c,\ \Psi_c)
	$$
- Marginally:  
	$$
	Y_j \sim \sum_{c=1}^C \rho_c\, \mathcal{N}(\nu_c,\Psi_c)
	$$
    a **Gaussian mixture** over disease modes.

We define $g_0$ as a fixed multilayer perceptron (MLP):
	$$
	\begin{align}
	H_j^{(Y,1)} &= \sigma(Y_j C_1 + b_1^{(Y)})\\
	H_j^{(Y,2)} &= \sigma(H_j^{(Y,1)} C_2 + b_2^{(Y)})\\
	\mu^{(W)}_j &= H_j^{(Y,2)} C_3 + b_3^{(Y)} \in \mathbb{R}^k
	\end{align}
	$$

All matrices $C_1, C_2, C_3$ and biases $b_1^{(Y)}, b_2^{(Y)}, b_3^{(Y)}$ are **fixed**, sampled once and kept constant. Prior for $W_j$:
$$W_j \mid Y \sim \mathcal{N}\big(\mu^{(W)}_j,\ \Sigma^{(W)}\big),\quad  \Sigma^{(W)} = \sigma_W^2 I_k.  $$
---

# Biological Story Justifying Each Component of the Generative Model

You want to simulate a matrix  
$$
R_{ij} = \text{“relevance of gene (i) to disease (j)”}
$$  
together with gene features $X$, disease features $Y$, and a gene graph $G$, in a way that reflects how real-world biology works.

We assume that a disease is activated by a combination of gene effects and that a gene contributes to many diseases to varying degrees. The bilinear term captures this structure: it reflects **epistatic interactions** between genes (through shared latent factors in $U$) and **modular disease architecture** (through pathway-level representations in $W$).

To connect this continuous mechanistic layer with discrete causal annotations, we introduce a  **latent continuous causal activation variable**. This represents the underlying biological strength by which gene $i$ influences the pathway mechanisms of disease $j$. This variable models continuous molecular effects such as pathway alignment, regulatory impact, or functional compatibility. The binary association we observe arises by thresholding this latent causal signal. This marginalizes to the probit model:
$$
R_{ij} \mid U_i, W_j \sim \text{Bernoulli}\left( \Phi\left( \dfrac{U_i^\top W_j + b}{\sigma_Z} \right) \right)
$$


**Genes**

Real biological processes are governed by **latent functional modules** or pathways:
- genes cluster into groups with similar cellular roles,
- membership in these groups is not directly observed,
- gene activity and interactions arise from these hidden functional axes.

Thus we assume each gene $i$ has a **latent functional state**: $H_i \in \mathbb{R}^L$. These states are **not homogeneous** across genes. Instead, biology is structured:
- genes belong to families,
- pathways impose correlated behavior,
- evolutionary forces create groups of functionally similar genes.

This motivates modeling $H_i$ through a **Gaussian mixture**. Each mixture component models a **gene module** or **pathway category**. $D$ is the number of biological modules in your synthetic system.

In real datasets:
- expression profiles,
- gene sequence embeddings,
- GO annotations,
are **noisy reflections** of deeper biological signals. Thus you generate gene features as $X_i$ a noisy high-dimensional signature of $H_i$.

Protein–protein interaction networks (PPIs), gene co-expression graphs, and regulatory networks exhibit a key property:
*"Genes that participate in similar biological functions tend to be connected."*

In your model, this is encoded by generating edges according to distance in $H$. Genes with similar latent states are more likely to interact. This mirrors actual PPI topology, which is highly **clustered** and **functionally modular**. The graph is a _second view_ of biological structure, complementary to $X$.

Each gene’s contribution to disease is not determined solely by its own features. Instead:
- genes act in pathways,
- disease mechanisms involve coordinated gene activity,
- network effects propagate biological signals (e.g., diffusion of dysfunction).

$f_0(X,G)$ corresponds biologically to:
- Signal propagation over the PPI/regulatory network.
- Combining local gene features with neighbors.
- A non-linear transformation (e.g. via ReLU/tanh) reflecting complex biological activation patterns.

**Diseases**

Diseases can be grouped into **etiological classes**, such as:
- inflammatory,
- metabolic,
- oncological,
- neurodegenerative

Thus you generate disease features using a mixture $Y_j \mid z_j^Y$ of $C$ disease categories. $Y_j$ might represent phenotype embeddings, symptom vectors, or curated ontologies.

Just as genes require network context, diseases exhibit non-linear interactions between phenotypic dimensions:
- different symptom dimensions combine non-linearly,
- comorbidities reflect interactions across disease features,
- clinical descriptors interact (e.g., severity × duration effects).

Thus $g_0(Y)$ models multi-dimensional, non-linear disease mechanisms.
