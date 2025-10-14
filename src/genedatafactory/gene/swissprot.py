import gzip
from typing import List, Set
import pandas as pd
from Bio import SwissProt


def _extract_gene_ids(rec) -> Set[int]:
    """Extract NCBI Gene IDs from a SwissProt record.

    Args:
        rec: A Bio.SwissProt.Record object.

    Returns:
        Set[int]: Set of NCBI Entrez Gene IDs referenced by the record.
    """
    gene_ids: Set[int] = set()
    for db, *rest in rec.cross_references:
        if db == "GeneID" and rest:
            try:
                gene_ids.add(int(rest[0]))
            except (TypeError, ValueError):
                continue
    return gene_ids


def _extract_keywords(rec) -> Set[str]:
    """Extract UniProt keywords from a SwissProt record.

    Args:
        rec: A Bio.SwissProt.Record object.

    Returns:
        Set[str]: Set of non-empty UniProt keywords.
    """
    return {kw for kw in (rec.keywords or []) if kw}


def read_swissprot(dat_gz_path: str, geneid: List[int]) -> pd.DataFrame:
    """Parse a Swiss-Prot file and extract UniProt keywords linked to given GeneIDs.

    Args:
        dat_gz_path (str): Path to the compressed UniProt Swiss-Prot data file (e.g., 'uniprot_sprot.dat.gz').
        geneid (List[int]): List of NCBI Gene IDs to retrieve keyword annotations for.

    Returns:
        pd.DataFrame: DataFrame with two columns:
            - **GeneID** (*int*): NCBI Gene identifier.
            - **Keyword** (*str*): UniProt keyword associated with that gene.
        Each (GeneID, Keyword) pair is unique.
    """
    target_geneids: Set[int] = set(int(g) for g in geneid)
    rows: List[list] = []

    with gzip.open(dat_gz_path, "rt", encoding="utf-8") as fh:
        for rec in SwissProt.parse(fh):
            rec_geneids = _extract_gene_ids(rec)
            matched = rec_geneids & target_geneids
            if not matched:
                continue

            kws = _extract_keywords(rec)
            if not kws:
                continue

            for gid in matched:
                for kw in kws:
                    rows.append([gid, kw])

    df = pd.DataFrame(rows, columns=["GeneID", "Keyword"]).drop_duplicates(
        ignore_index=True
    )
    return df
