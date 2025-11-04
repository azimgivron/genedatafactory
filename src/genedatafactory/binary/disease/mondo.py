import re
from collections import defaultdict
from typing import Dict, List, Set

import networkx as nx
import obonet
import pandas as pd

_MONDO_DAG = None
_DIGRAPH = None


def get_mapping() -> Dict[str, Set[str]]:
    """Build a mapping from OMIM (MIM) numbers to MONDO term IDs.

    Returns:
        Dict[str, Set[str]]: Mapping where each key is an OMIM ID (string of digits)
            and each value is a set of corresponding MONDO term identifiers.
    """
    omim_to_mondo = defaultdict(set)
    xref_re = re.compile(r"OMIM(P|PS)?:\d+")

    for node, data in _MONDO_DAG.nodes(data=True):
        for x in data.get("xref", []):
            if xref_re.match(x):
                mim = re.search(r"(\d+)", x).group(1)
                omim_to_mondo[int(mim)].add(node)
    return omim_to_mondo


def precompute_ancestor_closure() -> None:
    """Precompute the directed MONDO subgraph of 'is_a' relationships."""
    global _DIGRAPH
    is_a_edges = [
        (u, v)
        for u, v, d in _MONDO_DAG.edges(data=True)
        if d.get("relation") in (None, "is_a")  # unlabeled = is_a by default
    ]
    _DIGRAPH = nx.DiGraph()
    _DIGRAPH.add_edges_from(is_a_edges)


def ancestors_inclusive(node: str) -> Set[str]:
    """Return the inclusive set of ancestors for a MONDO term.

    Args:
        node (str): MONDO term identifier.

    Returns:
        Set[str]: The node and all its ancestors in the MONDO hierarchy.
    """
    return {node} | nx.ancestors(_DIGRAPH, node)


def mondo_binary_vector(
    disease_ids: List[str], omim_to_mondo: Dict[str, Set[str]]
) -> pd.DataFrame:
    """Generate a binary mapping of OMIM diseases to their MONDO ancestors.

    Args:
        disease_ids (List[str]): List of OMIM MIM numbers as strings.
        omim_to_mondo (Dict[str, Set[str]]): Mapping from OMIM IDs to MONDO term IDs.

    Returns:
        pd.DataFrame: DataFrame with columns ['MIM', 'TermID'],
            each row representing an active MONDO term for a given OMIM disease.
    """
    terms = [n for n in _DIGRAPH.nodes() if n.startswith("MONDO:")]
    term_index = {t: i for i, t in enumerate(sorted(terms))}

    data = []
    for mim in disease_ids:
        for mid in omim_to_mondo.get(mim, []):
            if mid in _DIGRAPH:
                for anc in ancestors_inclusive(mid):
                    j = term_index.get(anc)
                    if j is not None:
                        data.append((mim, j))

    df = pd.DataFrame(data, columns=["MIM number", "TermID"])
    return df.drop_duplicates()


def read_mondo(path: str, disease_ids: List[str]) -> pd.DataFrame:
    """Read MONDO ontology and produce OMIM-MONDO ancestor mappings.

    Args:
        path (str): Path to the MONDO ontology file (.obo format).
        disease_ids (List[str]): List of OMIM MIM numbers as strings.

    Returns:
        pd.DataFrame: Binary mapping between diseases and MONDO ancestor terms.
    """
    global _MONDO_DAG
    _MONDO_DAG = obonet.read_obo(path)
    omim_to_mondo = get_mapping()
    precompute_ancestor_closure()
    return mondo_binary_vector(disease_ids, omim_to_mondo)
