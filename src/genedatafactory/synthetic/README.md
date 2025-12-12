# Data Generation
## 0. Notation and goal
- $n$: number of genes (row entities).
- $m$: number of diseases (column entities).
- $k$: latent dimension of gene and disease factors.
- $d_X$: dimension of gene features $X_i \in \mathbb{R}^{d_X}$.
- $d_Y$: dimension of disease features $Y_j \in \mathbb{R}^{d_Y}$.
- $L_g$: dimension of latent embedding $H_i$.
- $L_d$: dimension of latent embedding $D_i$.
- $F$: number of **gene mixture components** (modes).
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
    z_i^X \sim \text{Categorical}(\pi_1,\dots,\pi_F),\quad \sum_{d=1}^F \pi_d = 1
    $$

Given $z_i^X$, we draw a latent embedding $H_i \in \mathbb{R}^{L_g}$:
	$$
	H_i \mid z_i^X \sim \mathcal{N}(\tilde{\mu}_d,\ \tilde{\Sigma}_d),\quad d = 1,\dots,F
	$$
- Marginally over $z_i^X$, $H_i$ is drawn from a **Gaussian mixture**:  
	$$H_i \sim \sum_{d=1}^F \pi_d\, \mathcal{N}(\tilde{\mu}_d,\tilde{\Sigma}_d)$$

Observed gene features $X_i \in \mathbb{R}^{d_X}$ are a noisy linear readout of $H_i$:
	$$X_i \mid H_i \sim \mathcal{N}(A_X H_i,\ \sigma_X^2 I_{d_X}),\quad A_X \in \mathbb{R}^{d_X \times L_g}$$

Because $H_i$ is a Gaussian mixture, the marginal distribution of $X_i$ is a **Gaussian mixture**.

We build an undirected graph $G = (V,E)$ on the genes, with $V = {1,\dots,n}$, adjacency matrix $A = (A_{i\ell})$.

For each unordered pair $(i,\ell)$, $i<\ell$:
$$
A_{i\ell} \mid H_i, H_\ell \sim \text{Bernoulli}(p_{i\ell}),\quad  
p_{i\ell} = \exp\left(-\frac{\|H_i - H_\ell\|^2}{2\eta^2}\right) 
$$
for some scale parameter $\eta > 0$. This defines a **random geometric graph** in latent space.

## 3. Prior for $U$ via $f(X,G)$

We want $U_i$ to be a **function** of both $X$ and the gene graph $G$, while keeping the conditional distribution Gaussian. For each gene $i$:  
$$
\mu^{(U)}_i = [f(H,G)]_i \in \mathbb{R}^k
$$
We define:  
$$U_i \mid H,G \sim \mathcal{N}\big(\mu^{(U)}_i,\ \Sigma^{(U)}\big),\quad  
\Sigma^{(U)} = \sigma_U^2 I_k.  $$

## 4. Disease side (columns): mixture features and linear $g_0(Y)$

We now specify the disease side. We generate disease features $Y$ from a latent embedding $D$. 

For each disease $j \in {1,\dots,m}$:
- **Mixture assignment (discrete):**  
	$$z_j^Y \sim \text{Categorical}(\rho_1,\dots,\rho_C),\quad \sum_{d=1}^C \rho_d = 1$$

Given $z_j^Y = c$, we draw a latent embedding $D_j \in \mathbb{R}^{L_d}$:
	$$
	D_j \mid z_j^Y \sim \mathcal{N}(\tilde{\mu}_c,\ \tilde{\Sigma}_c),\quad c = 1,\dots,C
	$$
- Marginally over $z_j^Y$, $D_j$ is drawn from a **Gaussian mixture**:  
	$$D_j \sim \sum_{c=1}^C \rho_c\, \mathcal{N}(\tilde{\mu}_c,\tilde{\Sigma}_c)$$
    a **Gaussian mixture** over disease modes.

Observed disease features $Y_i \in \mathbb{R}^{d_Y}$ are a noisy linear readout of $D_i$:
	$$Y_i \mid D_i \sim \mathcal{N}(A_Y D_i,\ \sigma_Y^2 I_{d_Y}),\quad A_Y \in \mathbb{R}^{d_Y \times L_d}$$

Because $D_i$ is a Gaussian mixture, the marginal distribution of $Y_i$ is a **Gaussian mixture**.

For each disease $g$:  
$$
\mu^{(W)}_j = g(D)_j \in \mathbb{R}^k
$$
We define:  
$$W_j \mid D \sim \mathcal{N}\big(\mu^{(W)}_j,\ \Sigma^{(W)}\big),\quad  \Sigma^{(W)} = \sigma_W^2 I_k$$

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

Real biological processes are governed by **latent functional modules** or pathways:

* genes cluster into groups with similar cellular roles,
* membership in these groups is not directly observed,
* gene activity and interactions arise from these hidden functional axes.

Thus each gene $i$ has a **latent functional state** $H_i \in \mathbb{R}^{L_g}$. These states are **heterogeneous** across genes and organized into functional groups. A **Gaussian mixture** prior encodes this structure: each mixture component corresponds to a biological module or functional class. In real datasets, high-dimensional measurements—expression profiles, sequence-derived embeddings, GO annotations—represent **noisy observations** of deeper biological signals. This motivates generating gene features as
$$
X_i \approx A_X H_i + \text{noise}
$$

Protein–protein interaction networks, co-expression networks, and regulatory graphs share a key biological property:
> Genes involved in similar functions tend to be connected.

In the model, an edge probability that decays with latent distance $\|H_i - H_\ell\|$ produces a random geometric graph that mimics functional modularity of real PPIs. Thus the graph $G$ provides a **complementary view** of the same underlying biology captured in $X$. Finally, gene–disease relevance depends not only on intrinsic gene properties but also on **network context**. Pathways integrate signals across interacting genes. The mapping
$$
f(H,G)
$$
captures propagation of functional information through the network.

**Diseases**

Diseases also exhibit latent structure. Many can be grouped into **etiological clusters**:

* inflammatory,
* metabolic,
* oncological,
* neurodegenerative,
* infectious.

Such categories motivate a Gaussian mixture prior on the disease latent states $D_j$. The observed disease features $Y_j$ (symptom vectors, phenotype embeddings, ontology-derived features) are interpreted as noisy linear projections of these latent disease factors. The downstream mapping $$g(D_j)$$ produces the mean of $W_j$.

---

# Additional Note: $H$ and $D$ Are Latent, While $X$ and $Y$ Are Observed

The latent embeddings $H_i$ (for genes) and $D_j$ (for diseases) represent the true but **unobserved** biological states:

* functional gene modules,
* latent pathway activities,
* etiological disease classes.

In contrast, the feature matrices $X$ and $Y$ are **observable** but **noisy**. They provide incomplete, biased, and corrupted projections of $H$ and $D$:

* $X_i$ is a noisy expression/annotation-based reflection of $H_i$,
* $Y_j$ is a noisy phenotype/ontology-based reflection of $D_j$.

Thus, in downstream inference, **only $X$, $Y$, the graph $G$, and the binary matrix $R$ are available**. The true generative variables $H$ and $D$ remain hidden, forcing any model to learn from imperfect proxies—mirroring real biological data analysis, where the underlying biological mechanisms are never directly observed.
