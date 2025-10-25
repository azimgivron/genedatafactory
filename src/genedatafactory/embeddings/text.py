from typing import List

import nltk
import numpy as np
import pandas as pd

from genedatafactory.embeddings.biobert import BioBERTMeanEncoder

# NLTK sentence tokenizer
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def prepare_text(description: str) -> str:
    """Normalize, sentence-split, and deduplicate a gene description.

    Args:
      description: Raw concatenated text (e.g., concatenated GeneRIF sentences).

    Returns:
      A cleaned, deduplicated description string suitable for embedding.
    """
    from nltk.tokenize import sent_tokenize

    sents = [s.strip() for s in sent_tokenize(description) if s.strip()]
    uniq = list(dict.fromkeys(sents))  # preserve first occurrence order
    return " ".join(uniq)


def embed(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Generate BioBERT mean-pooled embeddings for text descriptions.

    Encodes each entry in the DataFrame's "description" column using
    BioBERT mean pooling and appends the resulting embedding vectors.

    Args:
        df (pd.DataFrame): Input DataFrame containing a "description" column.
        key (str): Column name identifying each record (e.g., "GeneID" or "MIM number").

    Returns:
        pd.DataFrame: DataFrame with the identifier column and embedding features:
            - key: Identifier from the input DataFrame.
            - embedding_0 ... embedding_767: Float vector components.
    """
    encoder = BioBERTMeanEncoder()
    embeddings = encoder.encode(df["description"].tolist())
    emb_array = embeddings.detach().cpu().numpy().astype(np.float32)
    emb_cols = [f"embedding_{i}" for i in range(emb_array.shape[1])]  # 0..767
    emb_df = pd.DataFrame(emb_array, columns=emb_cols)
    df_out = pd.concat([df[[key]].reset_index(drop=True), emb_df], axis=1)
    return df_out


def read_generifs_basic(path: str, gene_idss: List[int]) -> pd.DataFrame:
    """Load NCBI GeneRIFs, build unique descriptions per GeneID, and embed them.

    This function reads the `generifs_basic.gz` file from the NCBI Gene FTP,
    filters rows to the provided GeneIDs, aggregates GeneRIF sentences into a
    unique description per gene, and produces BioBERT mean-pooled embeddings.

    Args:
      path: Filesystem path to `generifs_basic.gz`.
      gene_idss: List of GeneIDs to retain.

    Returns:
      DataFrame with columns:
        - "GeneID" (int)
        - "embedding_0" ... "embedding_767": float vector components
    """
    names = [
        "Tax ID",
        "GeneID",
        "PubMed ID list",
        "last update timestamp",
        "GeneRIF text",
    ]
    usecols = ["GeneID", "GeneRIF text"]
    dtype = {"GeneID": "int32", "GeneRIF text": "string"}

    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        comment="#",
        header=None,
        names=names,
        usecols=usecols,
        dtype=dtype,
        engine="c",
    )

    # Filter target genes and drop empty GeneRIFs
    df = df[df["GeneID"].isin(gene_idss)].dropna(subset=["GeneRIF text"])

    # Build unique description per GeneID, then embed
    rif_text = df["GeneRIF text"].str.strip()
    df = (
        df.assign(_rif=rif_text)
        .groupby("GeneID", sort=False)["_rif"]
        .apply(lambda s: " ".join(pd.unique(s)))
        .reset_index(name="description")
    )
    df["description"] = df["description"].apply(prepare_text)
    embeded = embed(df, "GeneID")

    return embeded


def read_medgen_definitions(
    mgdef_path: str, mappings_path: str, disease_ids: List[int]
) -> pd.DataFrame:
    """Extract and embed disease descriptions for OMIM IDs using MedGen data.

    This function reads the MedGen definition file (`MGDEF.RRF.gz`) and the
    MedGen ID mapping file (`MedGenIDMappings.txt.gz`) to collect textual
    disease definitions corresponding to a set of OMIM (MIM) identifiers.
    It merges, cleans, and deduplicates the definitions, applies text
    preprocessing with `prepare_text()`, and then computes BioBERT embeddings
    using the `embed()` function.

    Args:
        mgdef_path (str): Path to the MedGen definitions file (`MGDEF.RRF.gz`),
            containing concept identifiers (CUI) and definition text.
        mappings_path (str): Path to the MedGen ID mapping file
            (`MedGenIDMappings.txt.gz`), linking OMIM identifiers to CUIs.
        disease_ids (List[int]): List of OMIM (MIM) disease identifiers to
            include in the output.

    Returns:
        pd.DataFrame: DataFrame containing BioBERT embeddings for each OMIM
        disease entry, with columns:
            - "MIM number": MIM number corresponding to the disease
            - "embedding_0" ... "embedding_767": float vector components
    """
    names = ["CUI", "DEF", "SOURCE", "SUPPRESS"]
    usecols = ["CUI", "DEF", "SOURCE"]
    dtype = {"CUI": "string", "DEF": "string"}
    df_def = pd.read_csv(
        mgdef_path,
        sep="|",
        lineterminator="\n",
        compression="gzip",
        comment="#",
        header=None,
        names=names,
        dtype=dtype,
        usecols=usecols,
        engine="c",
        index_col=False,
    )

    names = ["CUI_or_CN_id", "pref_name", "source_id", "source"]
    usecols = ["CUI_or_CN_id", "source_id", "source"]
    dtype = {"CUI_or_CN_id": "string", "source_id": "string", "source": "string"}
    df_map = pd.read_csv(
        mappings_path,
        sep="|",
        compression="gzip",
        comment="#",
        header=None,
        names=names,
        dtype=dtype,
        usecols=usecols,
        engine="c",
        index_col=False,
    )

    df_map = df_map[df_map["source"] == "OMIM"]
    df_map["MIM number"] = df_map["source_id"].astype(int)
    df = df_map.merge(df_def, left_on="CUI_or_CN_id", right_on="CUI", how="left")

    df = df[["MIM number", "DEF"]].rename(columns={"DEF": "description"})
    df = df.drop_duplicates().dropna()
    df = df[df["MIM number"].isin(disease_ids)]
    df = (
        df.groupby("MIM number", sort=False)["description"]
        .apply(lambda s: " ".join(pd.unique(s)))
        .reset_index(name="description")
    )
    df["description"] = df["description"].apply(prepare_text)
    embeded = embed(df, "MIM nnumber")

    return embeded
