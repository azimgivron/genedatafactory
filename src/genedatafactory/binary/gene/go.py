"""
Build a GO feature vector for a human gene (by NCBI GeneID).
- Source ontology:   go-basic.obo (Gene Ontology);
- Source annotations: NCBI gene2go (human rows only, taxon:9606);
The feature vector is a binary numpy array aligned to a stable GO vocabulary
(lexicographically sorted GO terms observed for human in gene2go).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from goatools.obo_parser import GODag

_GO_DAG = None
_TERM_INDEX = None


def load_gene2go_human(path_gz: str) -> pd.DataFrame:
    """Load and parse the human subset of the NCBI gene2go.gz file.

    This function efficiently reads the compressed gene2go dataset from NCBI,
    filtering relevant columns and enforcing fixed data types for consistency.

    Args:
        path_gz (str): Path to the compressed `gene2go.gz` file.

    Returns:
        pd.DataFrame: A DataFrame containing gene-to-GO mappings with columns:
            - "GeneID" (int): NCBI Gene identifier.
            - "GO_ID" (str): Gene Ontology term identifier.
            - "Evidence" (str): Evidence code supporting the annotation.
            - "Category" (str): GO category (e.g., BP, MF, CC).
    """
    # The file’s header line in gene2go is commented (#), so we provide names explicitly.
    names = [
        "#Tax_ID",
        "GeneID",
        "GO_ID",
        "Evidence",
        "Qualifier",
        "GO_term",
        "PubMed",
        "Category",
    ]
    usecols = ["#Tax_ID", "GeneID", "GO_ID", "Evidence", "Category"]
    dtype = {
        "#Tax_ID": "int32",
        "GeneID": "int32",
        "GO_ID": "string",
        "Evidence": "string",
        "Category": "string",
    }

    df = pd.read_csv(
        path_gz,
        sep="\t",
        compression="gzip",
        comment="#",
        header=None,
        names=names,
        usecols=usecols,
        dtype=dtype,
        engine="c",
    )  # type: ignore
    df["GO_ID"] = df["GO_ID"].str.strip()
    return df[["GeneID", "GO_ID", "Evidence", "Category"]]


def _init_worker(go_obo_path: str, term_index: dict[str, int]):
    """Initialize worker process with GO DAG and term index.

    Args:
        go_obo_path (str): Path to the GO OBO file.
        term_index (dict[str, int]): Mapping from GO term to column index.
    """
    global _GO_DAG, _TERM_INDEX
    _GO_DAG = GODag(go_obo_path, optional_attrs={"relationship"})
    _TERM_INDEX = term_index


def _ancestors_including_self(term: str) -> set[str]:
    """Return the set of ancestor GO terms including the term itself.

    Args:
        term (str): GO term ID.

    Returns:
        set[str]: Ancestor GO terms including the input term.
    """
    if term in _GO_DAG:
        return {term} | set(_GO_DAG[term].get_all_upper())
    return {term}


def _gene_to_col_indices_worker(gene: int, terms: set[str]) -> tuple[int, np.ndarray]:
    """Map a gene to column indices based on its GO terms and ancestors.

    Args:
        gene (int): Gene identifier.
        terms (set[str]): Set of directly annotated GO terms.

    Returns:
        tuple[int, np.ndarray]: Gene ID and array of corresponding column indices.
    """
    cols = set()
    for t in terms:
        for a in _ancestors_including_self(t):
            j = _TERM_INDEX.get(a)
            if j is not None:
                cols.add(j)
    arr = np.fromiter(sorted(cols), dtype=np.int32) if cols else np.empty(0, np.int32)
    return gene, arr


def make_feature_vector(
    go_obo_path,
    gene_ids: List[int],
    gene2go_df: pd.DataFrame,
    vocab: List[str],
) -> pd.DataFrame:
    """Construct a sparse binary matrix linking genes to GO terms with propagation.

    Args:
        go_obo_path (str): Path to GO OBO file.
        gene_ids (List[int]): List of NCBI GeneIDs (rows).
        gene2go_df (pd.DataFrame): DataFrame containing 'GeneID' and 'GO_ID' columns.
        vocab (List[str]): List of GO terms (columns).

    Returns:
        pd.DataFrame: Sparse gene-GO mapping DataFrame.
    """
    # Precompute: mapping GO term -> vocab column index
    term_index: Dict[str, int] = {t: j for j, t in enumerate(vocab)}

    # Group direct GO terms per requested gene
    df = gene2go_df[gene2go_df["GeneID"].isin(gene_ids)]

    grouped: Dict[int, Set[str]] = (
        df.groupby("GeneID")["GO_ID"].apply(lambda s: set(s.dropna())).to_dict()
    )

    # Ensure every requested gene appears (possibly with empty set)
    for g in gene_ids:
        grouped.setdefault(g, set())

    # Run workers
    data = []
    with ProcessPoolExecutor(
        initializer=_init_worker, initargs=(go_obo_path, term_index)
    ) as ex:
        futures = [
            ex.submit(_gene_to_col_indices_worker, g, grouped[g]) for g in gene_ids
        ]
        for fut in as_completed(futures):
            g, cols = fut.result()
            data.extend([[g, voc_id] for voc_id in cols])
    sparse_df = pd.DataFrame(data, columns=["GeneID", "TermID"])
    return sparse_df.drop_duplicates()


def read_go(GO_path: str, gene2GO_path: str, gene_ids: List[int]) -> pd.DataFrame:
    """Generate a GO feature matrix for a set of genes.

    Args:
        GO_path (str): Path to GO OBO file.
        gene2GO_path (str): Path to NCBI gene2go.gz file.
        gene_ids (List[int]): List of target NCBI GeneIDs.

    Returns:
        pd.DataFrame: Sparse gene-GO mapping DataFrame.
    """
    gene2go = load_gene2go_human(gene2GO_path)
    vocab = sorted(gene2go["GO_ID"].unique())
    return make_feature_vector(
        GO_path,
        gene_ids=gene_ids,
        gene2go_df=gene2go,
        vocab=vocab,
    )
