import pandas as pd


def read_omim(path: str) -> pd.DataFrame:
    """Parse an OMIM gene-phenotype mapping file into a DataFrame.

    This function reads a tab-separated OMIM file (e.g., `mim2gene_medgen`)
    while skipping metadata lines beginning with '#'. It infers column names
    from the first non-comment line, then filters to phenotype entries
    linked to a valid GeneID and without comments.

    Args:
        path (str): Path to the OMIM TSV file (e.g., 'mim2gene_medgen').

    Returns:
        Tuple[pd.DataFrame, Set[int]]: Tuple with:
            - Filtered DataFrame containing only human phenotype
            associations with valid GeneIDs. Columns are
            derived from the file header.
            - The set of existing Genes.
    """
    orign = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=["MIM number", "GeneID", "type", "Source", "MedGenCUI", "Comment"],
        dtype={
            "MIM number": "int32",
            "GeneID": "string",
            "type": "string",
            "Source": "string",
            "MedGenCUI": "string",
            "Comment": "string",
        },
    )

    # Filter to phenotype entries with valid GeneIDs and
    # remove non-disease, susceptibility, question and QTL, so keep confirmed
    df = orign[
        (orign["type"] == "phenotype")
        & (orign["GeneID"] != "-")
        & (orign["Comment"] == "-")
    ]
    df = df[["GeneID", "MIM number"]]
    df["MIM number"] = df["MIM number"].astype("int32")
    df["GeneID"] = df["GeneID"].astype("int32")
    df = df.drop_duplicates()
    return df, set(orign[orign["GeneID"] != "-"]["GeneID"].dropna().astype("int32"))
