from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def count(
    genes: Set[int],
    diseases: Set[int],
    others: List[pd.DataFrame],
    gene_key: str = "GeneID",
    disease_key: str = "MIM number",
) -> Tuple[Set[int], Set[int]]:
    """Compute intersecting gene and disease identifiers across multiple DataFrames.

    Args:
        genes (Set[int]): Set of gene IDs.
        diseases (Set[int]): Set of disease IDs.
        others (List[pd.DataFrame]): Other DataFrames to intersect with `main_df`.
        gene_key (str, optional): Column name for gene identifiers. Defaults to "GeneID".
        disease_key (str, optional): Column name for disease identifiers. Defaults to "MIM number".

    Returns:
        Tuple[Set[int], Set[int]]: Intersected sets of gene IDs and disease IDs.
    """
    for df in others:
        if gene_key in df.columns:
            genes = genes.intersection(set(df[gene_key].astype("int32")))
        elif disease_key in df.columns:
            diseases = diseases.intersection(set(df[disease_key].astype("int32")))
    return genes, diseases


def new_mapping(df: pd.DataFrame, key: str, mapping: Dict[int, int]) -> pd.DataFrame:
    """Replace identifiers in a DataFrame column using a provided mapping.

    Args:
        df (pd.DataFrame): Input DataFrame.
        key (str): Column name whose values will be remapped.
        mapping (Dict[int, int]): Dictionary mapping old IDs to new integer indices.

    Returns:
        pd.DataFrame: DataFrame with updated identifiers in the specified column.
    """
    df.loc[:, key] = df[key].astype("int32")
    df.loc[:, key] = df[key].map(lambda x: mapping[x])
    return df


def remap(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    """Reindex a numeric column by assigning new consecutive integer IDs.

    Args:
        df (pd.DataFrame): Input DataFrame.
        idx (int): Index position of the column to reindex.

    Returns:
        pd.DataFrame: DataFrame with the specified column remapped to consecutive integers.
    """
    index_set = set(df.iloc[:, idx].astype("int32"))
    mapping = {
        np.int32(old_id): np.int32(new_id) for new_id, old_id in enumerate(index_set)
    }
    df.iloc[:, idx] = df.iloc[:, idx].map(lambda x: mapping[x])
    return df
