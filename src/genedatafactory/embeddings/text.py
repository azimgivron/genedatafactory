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


def unique_description(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GeneRIF sentences into one unique description per GeneID.

    Expects a DataFrame with columns:
      - "GeneID" (int)
      - "GeneRIF text" (str)

    Args:
      df: Input DataFrame with columns ["GeneID", "GeneRIF text"].

    Returns:
      A DataFrame with columns:
        - "GeneID"
        - "description" (clean, deduplicated text per GeneID)
    """
    rif_text = df["GeneRIF text"].str.strip()
    df_out = (
        df.assign(_rif=rif_text)
        .groupby("GeneID", sort=False)["_rif"]
        .apply(lambda s: " ".join(pd.unique(s)))
        .reset_index(name="description")
    )
    df_out["description"] = df_out["description"].apply(prepare_text)
    return df_out


def embed(df: pd.DataFrame) -> pd.DataFrame:
    """Embed gene descriptions with BioBERT (mean pooling) and attach vectors.

    Args:
      df: DataFrame with a "description" column (one row per GeneID).

    Returns:
      A DataFrame with columns:
        - "GeneID"
        - "embedding_0": float
        ...
        - "embedding_767": float
    """
    encoder = BioBERTMeanEncoder()
    embeddings = encoder.encode(df["description"].tolist())
    emb_array = embeddings.detach().cpu().numpy().astype(np.float32)
    emb_cols = [f"embedding_{i}" for i in range(emb_array.shape[1])]  # 0..767
    emb_df = pd.DataFrame(emb_array, columns=emb_cols)
    df_out = pd.concat([df[["GeneID"]].reset_index(drop=True), emb_df], axis=1)
    return df_out


def read_generifs_basic(path: str, geneids: List[int]) -> pd.DataFrame:
    """Load NCBI GeneRIFs, build unique descriptions per GeneID, and embed them.

    This function reads the `generifs_basic.gz` file from the NCBI Gene FTP,
    filters rows to the provided GeneIDs, aggregates GeneRIF sentences into a
    unique description per gene, and produces BioBERT mean-pooled embeddings.

    Args:
      path: Filesystem path to `generifs_basic.gz`.
      geneids: List of GeneIDs to retain.

    Returns:
      DataFrame with columns:
        - "GeneID" (int)
        - "embedding_0": BioBERT mean-pooled 0th dimension of the embedding.
        ...
        - "embedding_767": BioBERT mean-pooled 767th dimension of the embedding.
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
    df = df[df["GeneID"].isin(geneids)].dropna(subset=["GeneRIF text"])

    # Build unique description per GeneID, then embed
    df = unique_description(df)
    embeded = embed(df)

    return embeded
