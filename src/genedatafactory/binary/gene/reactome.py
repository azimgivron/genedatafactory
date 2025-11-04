from typing import List

import pandas as pd


def read_reactome(path: str, gene_ids: List[int]) -> pd.DataFrame:
    """Read and filter Reactome NCBI2Reactome mapping file.

    Args:
        path (str): Path to the Reactome mapping file (e.g., 'NCBI2Reactome.txt').
        gene_ids (List[int]): List of NCBI Gene IDs to retain.

    Returns:
        pd.DataFrame: Filtered DataFrame containing curated Homo sapiens
            gene-pathway associations limited to the provided GeneIDs.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["GeneID", "PathwayName", "PathwayID", "URL", "Evidence", "Species"],
        dtype={
            "GeneID": "string",
            "PathwayName": "string",
            "PathwayID": "string",
            "URL": "string",
            "Evidence": "string",
            "Species": "string",
        },
    )

    # Keep only Homo sapiens and curated entries
    curated_codes = {"TAS", "IDA", "EXP"}  # manually curated evidence codes
    df_curated = df[
        (df["Species"] == "Homo sapiens")
        & (df["Evidence"].isin(curated_codes))
        & (df["GeneID"].isin(pd.Series(gene_ids).astype(str)))
    ].dropna()

    df_curated["GeneID"] = pd.to_numeric(
        df_curated["GeneID"], downcast="integer", errors="coerce"
    )
    df_curated = df_curated.dropna()
    mapping = {
        t: i for i, t in enumerate(df_curated["PathwayID"].drop_duplicates().to_list())
    }
    df_curated["PathwayID"] = df_curated["PathwayID"].map(lambda x: mapping[x])

    df_curated = df_curated[["GeneID", "PathwayID"]]
    return df_curated.drop_duplicates()
