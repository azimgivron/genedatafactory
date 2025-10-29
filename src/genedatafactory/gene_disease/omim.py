import pandas as pd


def read_omim(path: str) -> pd.DataFrame:
    """Parse an OMIM gene–phenotype mapping file into a DataFrame.

    This function reads a tab-separated OMIM file (e.g., `mim2gene_medgen`)
    while skipping metadata lines beginning with '#'. It infers column names
    from the first non-comment line, then filters to phenotype entries
    linked to a valid GeneID and without comments.

    Args:
        path (str): Path to the OMIM TSV file (e.g., 'mim2gene_medgen').

    Returns:
        pd.DataFrame: Filtered DataFrame containing only human phenotype
        associations with valid GeneIDs. Columns are derived from the file header.
    """
    df = pd.read_csv(
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

    # Filter to phenotype entries with valid GeneIDs and no comment
    df = df[
        (df["type"] == "phenotype") & (df["GeneID"] != "-") & (df["Comment"] == "-")
    ]
    df["GeneID"] = df["GeneID"].astype("int32")

    return df[["GeneID", "MIM number"]]
