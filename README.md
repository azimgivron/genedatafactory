# What is Coming

We will evaluate and implement the following improvements:

- [ ] Enhance data utilization by leveraging:
  - [ ] Ontology structures using tools such as [mOWL](https://github.com/bio-ontology-research-group/mowl)
  - [ ] Knowledge graph embedding frameworks like [PyKEEN](https://github.com/pykeen/pykeen)
  - [ ] [ProteinBERT](https://github.com/nadavbra/protein_bert) for protein sequence data
  - [x] [BioBERT](https://github.com/dmis-lab/biobert) to embed OMIM gene and disease descriptions
* Extend beyond the current use of the STRING graph by incorporating additional curated data sources through tools such as [BioKG](https://github.com/dsi-bdi/biokg), which offers an elegant and integrated way to include more biological knowledge.


# 🧬 Genedatafactory

**Dataset generation and preprocessing tools for gene prioritization tasks**

## 📖 Overview

`genedatafactory` is a Python package designed to automate the download, integration, and preprocessing of large-scale biomedical datasets used in **gene prioritization** experiments.
It standardizes the extraction of information from public databases (NCBI, GO, Reactome, STRING) to produce harmonized data matrices ready for downstream machine learning.

This package forms the first stage of a complete **gene prioritization pipeline** — transforming raw biological data into structured numerical representations.

## ⚙️ Features

* 📥 **Automatic download** of reference biomedical datasets (ClinVar, OMIM, GO, UniProt, etc.)
* 🧩 **Data integration** across genes, diseases, and ontology resources
* 🧠 **Preprocessing utilities** for graph- and matrix-based representations
* 🧾 **Reproducible exports** to CSV for downstream modeling
* 🔗 **Gene network access** from PPI graph construction
* 🧪 Designed for research in **bioinformatics**, **genetics**, and **machine learning**

## 🧰 Installation

For *users*:

```bash
pip install git+https://github.com/azimgivron/genedatafactory.git
```

or for *developers*:

```bash
git clone https://github.com/azimgivron/genedatafactory.git
cd genedatafactory
pip install -e .
```

## 🚀 Usage

Once installed, the main data generation process can be launched from the command line:

```bash
genedatafactory -i <input_folder> -o <output_folder>
```

**Example:**

```bash
genedatafactory -i ./data/raw -o ./data/processed
```

### 🧩 What it does

1. Checks for required datasets (downloads missing files).
2. Processes OMIM, GO, HPO, SwissProt, Reactome, Mondo, STRING, and ClinVar data.
3. Saves integrated and cleaned tables as CSV files in the output folder.

## 📦 Directory structure

```
src/
└── genedatafactory/
├── _version.py
├── config.yaml
│
├── binary/
│   ├── disease/
│   │   ├── hpo.py              # HPO-based binary feature extraction
│   │   └── mondo.py            # MONDO ontology-based binary features
│   │
│   └── gene/
│       ├── go.py               # Gene Ontology binary encoding
│       ├── reactome.py         # Reactome pathway binary encoding
│       └── swissprot.py        # SwissProt annotation binary encoding
│
├── embeddings/
│   ├── biobert.py              # BioBERT embeddings for text data
│   ├── ontology.py             # Ontology-derived embeddings
│   ├── pathway.py              # Pathway-level embeddings
│   ├── sequence.py             # Sequence-based embeddings
│   └── text.py                 # Text cleaning and vectorization
│
├── gene_disease/
│   ├── clinvar.py              # ClinVar-based disease associations
│   └── omim.py                 # OMIM-based gene–disease associations
│
├── graph/
│   └── string_net.py           # STRING network integration and graph utils
│
└── script/
└── main.py                 # Main entry point for data encoding workflow
```

---

## 🧬 Data Sources and Their Roles in Gene Prioritization

The `genedatafactory` package integrates multiple **complementary biological databases**, each contributing unique evidence about gene function, disease mechanisms, or molecular interactions.
By combining these heterogeneous data types, the framework captures a **multifaceted view of gene–disease relationships**, improving the robustness and interpretability of prioritization models.

| Source                                         | Type                                       | Description                                                                                                                                                     | Complementarity & Benefit for Gene Prioritization                                                                                      |
| ---------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **OMIM (Online Mendelian Inheritance in Man)** | Curated gene–disease associations          | OMIM provides authoritative mappings between genes and Mendelian disorders. It is manually curated and considered the gold standard for confirmed associations. | Serves as **ground truth** for training and evaluation. Ensures high-confidence supervision for learning disease relevance.            |
| **HPO (Human Phenotype Ontology)**             | Disease-centric ontology                   | Describes phenotypic abnormalities observed in human diseases using standardized vocabulary terms organized in a hierarchical ontology.                         | Connects diseases through **shared phenotypic patterns**. Enables identification of genes associated with similar symptom profiles.    |
| **GO (Gene Ontology)**                         | Gene-centric ontology                      | Annotates genes with biological processes, molecular functions, and cellular components. Each term reflects experimental or computational evidence.             | Encodes **functional similarity** among genes. Useful for transferring disease relevance between genes with overlapping functions.     |
| **UniProt / SwissProt**                        | Protein function and structure annotations | High-quality protein information, including domain composition, subcellular localization, and enzymatic activities.                                             | Adds **biochemical and structural context** to gene function, enhancing model interpretability beyond GO.                              |
| **Reactome**                                   | Pathway database                           | Curated molecular pathways describing how gene products interact in biological processes.                                                                       | Captures **pathway-level co-involvement**, helping identify genes participating in the same mechanistic routes as known disease genes. |
| **MONDO (Monarch Disease Ontology)**           | Disease ontology                           | Integrates multiple disease classification systems (OMIM, Orphanet, DOID, etc.) into a unified ontology.                                                        | Provides a **hierarchical disease structure**, facilitating generalization between related disorders.                                  |
| **STRING**                                     | Protein–protein interaction network        | Aggregates experimental and predicted gene–gene associations based on co-expression, text mining, and database co-occurrence.                                   | Encodes **topological proximity** between genes. Helps infer potential disease genes through network diffusion and embedding methods.  |
| **ClinVar**                                    | Variant and clinical significance database | Contains clinically observed genetic variants linked to diseases, along with their interpretations (e.g., pathogenic, benign).                                  | Enables construction of a second gene disease association dataset based on variants.       |
| **NCBI Gene References Into Function** | Gene functional descriptions              | Short, literature-derived statements summarizing experimentally supported gene functions, curated from PubMed.                                                  | Provides **textual gene-level context** for embedding with BioBERT, enriching functional representations used in prioritization.       |
| **MedGen**                                     | Disease descriptions and concept mappings  | Integrates clinical and genetic disease concepts from OMIM, MeSH, Orphanet, and UMLS, providing standardized definitions and CUIs.                              | Supplies **concise disease definitions** for semantic embedding (via BioBERT), improving textual alignment between gene and disease concepts. |

### 🧠 Integration Rationale

Each dataset represents a **different dimension of biological knowledge**:

* **OMIM and ClinVar** provide *direct genetic evidence* of disease relevance.  
* **HPO and MONDO** describe *phenotypic and ontological relationships* among diseases.  
* **GO, SwissProt, and Reactome** capture *functional and mechanistic similarity* among genes.  
* **STRING** integrates *molecular interaction networks* supporting indirect association discovery.  
* **GENE_RIFs** offer *literature-derived gene function descriptions*, capturing experimental and contextual insights not found in structured annotations.  
* **MedGen** provides *standardized disease definitions* that enrich the semantic understanding of disorders, enabling effective text-based representation learning.

By harmonizing these sources, `genedatafactory` enables the creation of **comprehensive multi-view datasets**, crucial for downstream algorithms that combine **functional, topological, textual, and clinical features** to prioritize candidate disease genes effectively.


---

## 🧠 Output Files

All datasets are stored in a **sparse format**, meaning that only entries with non-zero values are explicitly listed. If a value column is not present, the corresponding entries are implicitly assigned a value of **1**. The overall number of genes and diseases is determined by the count of unique identifiers found in the `gene_disease.csv` file.

### 🧩 **Generated Datasets**

| Dataset                     | File name          | Description                                                                                                                                                                                                          |
| ---------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OMIM–Gene Map**            | `gene_disease.csv` | Gene–disease association table derived from OMIM (MIM2Gene). Each row represents a confirmed link between a gene (NCBI Gene ID) and a disease (OMIM number).                                                         |
| **HPO Annotations**          | `hpo.csv`          | Binary feature matrix for diseases (OMIM IDs) across all HPO terms. A value of **1** indicates that the disease is annotated with the term (including propagated ancestors in the ontology), and **0** otherwise.    |
| **GO Annotations**           | `go.csv`           | Binary feature matrix for genes (NCBI Gene IDs) across Gene Ontology (GO) terms. Encodes biological process, molecular function, and cellular component annotations, including propagated terms at all three levels. |
| **SwissProt Annotations**    | `swissprot.csv`    | Binary feature matrix for genes based on UniProt/SwissProt protein annotations. Similar to GO, each column represents a functional or structural protein feature linked to the gene.                                 |
| **Reactome Pathways**        | `reactome.csv`     | Binary feature matrix for genes across Reactome pathways. A value of **1** indicates that the gene participates in the pathway.                                                                                      |
| **Mondo Annotations**        | `mondo.csv`        | Binary feature matrix for diseases (OMIM IDs) across all MONDO terms.                                                                                                                                                |
| **STRING Network**           | `string.csv`       | Gene–gene interaction network derived from STRING database. Each edge represents a protein–protein interaction (PPI) with confidence scores provided by STRING.                                                      |
| **ClinVar Variant Network**  | `clinvar.csv`      | Gene–disease association table derived from ClinVar. Each record represents a validated association between a gene (NCBI Gene ID) and a disease (OMIM ID), annotated with the confidence or clinical significance level of the association. |
| **GENE_RIF Embeddings**      | `gene_rifs.csv`    | Text-based BioBERT embeddings of NCBI Gene descriptions extracted from GENE_RIFs (Gene References Into Function). Each row contains a gene identifier (NCBI Gene ID) and its 768-dimensional embedding vector.       |
| **MedGen Embeddings**        | `medgen.csv`       | Text-based BioBERT embeddings of disease definitions extracted from MedGen. Each row contains a disease identifier (OMIM number) and its 768-dimensional embedding vector.                                            |


### 🧠 **Conceptual Summary**

* **Feature matrices (HPO, GO, SwissProt, Reactome, Mondo, GENE_RIF Embeddings, MedGen Embeddings)**
  → Represent diseases or genes in a **vectorized format** suitable for machine learning and graph-based models.
  Each column corresponds to a controlled vocabulary term or ontology concept.

* **Graphs (STRING, ClinVar)**
  → Represent **relational structures** between genes and diseases, forming the backbone for graph-based learning tasks in gene prioritization.

## 🧩 Configuration

Dataset URLs and API endpoints are defined in the YAML configuration file at the root folder `config.yaml`.

## 🧑‍🔬 Citation

If you use this software in research, please cite:

> Azim Givron, *genedatafactory : A reproducible dataset generation toolkit for gene prioritization*, 2025.

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issue reports, and feature suggestions are welcome.
Please open a pull request or contact the maintainer if you wish to contribute improvements.

**Maintainer:** Azim Givron
📍 Brussels, Belgium
📧 [azim.givron@kuleuven.be](mailto:azim.givron@kuleuven.be)

