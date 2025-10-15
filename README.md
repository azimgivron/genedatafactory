# 🧬 Genedatafactory

**Dataset generation and preprocessing tools for gene prioritization tasks**

## 📖 Overview

`genedatafactory` is a Python package designed to automate the download, integration, and preprocessing of large-scale biomedical datasets used in **gene prioritization** experiments.  
It standardizes the extraction of information from public databases (NCBI, GO, reactome, String) to produce harmonized data matrices ready for downstream machine learning.

This package forms the first stage of a complete **gene prioritization pipeline** — transforming raw biological data into structured numerical representations.

## ⚙️ Features

- 📥 **Automatic download** of reference biomedical datasets (ClinVar, OMIM, GO, UniProt, etc.)
- 🧩 **Data integration** across genes, diseases, and ontology resources  
- 🧠 **Preprocessing utilities** for graph- and matrix-based representations  
- 🧾 **Reproducible exports** to CSV for downstream modeling  
- 🔗 **Gene and Disease network access** from PPI graph construction and ClinVar dataset
- 🧪 Designed for research in **bioinformatics**, **genetics**, and **machine learning**

## 🧰 Installation

For *users*:
```bash
pip install git+https://github.com/azimgivron/genedatafactory.git
```

or for *developpers*:
```bash
git clone https://github.com/azimgivron/genedatafactory.git
cd genedatafactory
pip install -e .
````

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
2. Processes OMIM, GO, HPO, SwissProt, Reactome, STRING, and ClinVar data.
3. Saves integrated and cleaned tables as CSV files in the output folder.


## 📦 Directory structure

```
src/
└── dataprocessing/
    ├── script/
    │   └── main.py           # Entry point for CLI command
    ├── __init__.py
    ├── gene_disease.py
    ├── go.py
    ├── hpo.py
    ├── swissprot.py
    ├── pathway.py
    ├── string_net.py
    ├── variant.py
    ├── mondo.py
    ├── config.yaml           # URLs and parameters for datasets
    └── _version.py           # Auto-generated version file
```

## 🧠 Output Files

All datasets are stored in a **sparse format**, meaning that only entries with non-zero values are explicitly listed. If a value column is not present, the corresponding entries are implicitly assigned a value of **1**. The overall number of genes and diseases is determined by the count of unique identifiers found in the `gene_disease.csv` file.


### 🧩 **Generated Datasets**

| Dataset                     | File name          | Description                                                                                                                                                                                                          |
| --------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OMIM–Gene Map**           | `gene_disease.csv` | Gene–disease association table derived from OMIM (MIM2Gene). Each row represents a confirmed link between a gene (NCBI Gene ID) and a disease (OMIM number).                                                         |
| **HPO Annotations**         | `hpo.csv`          | Binary feature matrix for diseases (OMIM IDs) across all HPO terms. A value of **1** indicates that the disease is annotated with the term (including propagated ancestors in the ontology), and **0** otherwise.    |
| **GO Annotations**          | `go.csv`           | Binary feature matrix for genes (NCBI Gene IDs) across Gene Ontology (GO) terms. Encodes biological process, molecular function, and cellular component annotations, including propagated terms at all three levels. |
| **SwissProt Annotations**   | `swissprot.csv`    | Binary feature matrix for genes based on UniProt/SwissProt protein annotations. Similar to GO, each column represents a functional or structural protein feature linked to the gene.                                 |
| **Reactome Pathways**       | `reactome.csv`      | Binary feature matrix for genes across Reactome pathways. A value of **1** indicates that the gene participates in the pathway.                                                                                      |
| **Mondo Annotations**       | `mondo.csv`      | Binary feature matrix for diseases (OMIM IDs) across all MONDO terms.|
| **STRING Network**          | `string.csv`       | Gene–gene interaction network derived from STRING database. Each edge represents a protein–protein interaction (PPI) with confidence scores provided by STRING.                                                      |
| **ClinVar Variant Network** | `clinvar.csv`      | Disease–disease network built from ClinVar. Two diseases are connected if they share at least one causal gene, with the edge weight reflecting the degree of shared genetic evidence.                                |

### 🧠 **Conceptual Summary**

* **Feature matrices (HPO, GO, SwissProt, Reactome)**
  → Represent diseases or genes in a **vectorized format** suitable for machine learning and graph-based models.
  Each column corresponds to a controlled vocabulary term or ontology concept.

* **Graphs (STRING, ClinVar)**
  → Represent **relational structures** between genes and diseases, forming the backbone for graph-based learning tasks in gene prioritization.

## 🧩 Configuration

Dataset URLs and API endpoints are defined in the YAML configuration file at the root folder ```config.yaml```.

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
