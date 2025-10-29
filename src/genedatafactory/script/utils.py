from typing import Dict, List, Set, Tuple

import pandas as pd


def count(
    main_df: pd.DataFrame,
    others: List[pd.DataFrame],
    gene_key: str = "GeneID",
    disease_key: str = "MIM number",
) -> Tuple[Set[int], Set[int]]:
    """_summary_

    Args:
        main_df (pd.DataFrame): _description_
        others (List[pd.DataFrame]): _description_
        gene_key (str, optional): _description_. Defaults to "GeneID".
        disease_key (str, optional): _description_. Defaults to "MIM number".

    Returns:
        Tuple[Set[int], Set[int]]: _description_
    """
    genes = set(main_df[gene_key])
    diseases = set(main_df[disease_key])

    for df in others:
        if gene_key in df.columns:
            genes &= set(df[gene_key].astype(int))
        elif disease_key in df.columns:
            diseases &= set(df[disease_key].astype(int))

    return genes, diseases


def new_mapping(df: pd.DataFrame, key: str, mapping: Dict[int, int]) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        key (str): _description_
        mapping (Dict[int, int]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df.loc[:, key] = df[key].astype(int)
    df.loc[:, key] = df[key].map(lambda x: mapping[x])
    return df


def remap(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        key (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    index_set = set(df[key].astype(int))
    mapping = {int(old_id): new_id for new_id, old_id in enumerate(index_set)}
    df.loc[:, key] = df[key].map(lambda x: mapping[x])
    return df
