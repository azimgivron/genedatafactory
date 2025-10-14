import time
from io import StringIO
from typing import List, Dict, Iterable

import numpy as np
import pandas as pd
import requests


def _chunks(lst: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _post_tsv(url: str, data: Dict, sleep_s: float = 0.3) -> pd.DataFrame:
    """POST to STRING API expecting TSV with header; returns empty DataFrame if no content."""
    r = requests.post(url, data=data, timeout=120)
    r.raise_for_status()
    txt = r.text.strip()
    if not txt:
        time.sleep(sleep_s)
        return pd.DataFrame()
    df = pd.read_csv(StringIO(txt), sep="\t")
    time.sleep(sleep_s)
    return df


def read_string(
    url_map: str,
    url_net: str,
    geneid: List[int],
    caller: str = "user",
    species: int = 9606,
    map_batch: int = 1000,
    net_batch: int = 1000,
    sleep_s: float = 0.3,
) -> pd.DataFrame:
    """Build a gene-gene weighted interaction network from STRING using batched API queries.

    This function maps NCBI Gene IDs to STRING protein IDs, retrieves interactions in
    manageable blocks to respect STRING API limits, and constructs an undirected edge list
    with normalized interaction confidence scores.

    Args:
        url_map (str): STRING API endpoint for ID mapping.
        url_net (str): STRING API endpoint for network retrieval.
        geneid (List[int]): List of NCBI Gene IDs to include in the network.
        caller (str, optional): Identifier string for API requests (e.g., project or user name).
        species (int, optional): NCBI taxon ID (default 9606 for human).
        map_batch (int, optional): Number of genes per mapping request (default 1000).
        net_batch (int, optional): Number of mapped STRING IDs per network block (default 1000).
        sleep_s (float, optional): Delay in seconds between API calls to avoid rate limits (default 0.3).

    Returns:
        pd.DataFrame: DataFrame with three columns:
            - GeneID_i (int): First gene.
            - GeneID_j (int): Second gene.
            - Weight (float): STRING combined confidence score.

    Notes:
        - Suitable for networks up to ~5,000 genes; for larger networks, use the
          downloadable STRING PPI files instead.
    """
    # MAP all NCBI → STRING (in batches)
    mappings = []
    for batch in _chunks(geneid, map_batch):
        params = {
            "species": species,
            "echo_query": 1,
            "limit": 1,
            "caller_identity": caller,
            "identifiers": "\r".join(map(str, batch)),
        }
        df = _post_tsv(url_map, params, sleep_s=sleep_s)
        if not df.empty:
            cols = [c for c in ["queryItem", "stringId"] if c in df.columns]
            mappings.append(df[cols].drop_duplicates())

    mapping = pd.concat(mappings, ignore_index=True).dropna().drop_duplicates()
    mapping["queryItem"] = mapping["queryItem"].astype(str)
    mapping["stringId"] = mapping["stringId"].astype(str)

    string_to_ncbi: Dict[str, str] = dict(
        zip(mapping["stringId"], mapping["queryItem"])
    )
    mapped_ncbi = set(mapping["queryItem"])

    # Prepare block lists of STRING IDs matching your NCBI set
    ncbi_sorted = sorted(mapped_ncbi, key=lambda x: int(x))
    ncbi_blocks = list(_chunks(ncbi_sorted, net_batch))
    string_blocks = []
    # Map each ncbi block to its string IDs
    ncbi_to_string = {v: k for k, v in string_to_ncbi.items()}  # NCBI → STRING
    for nb in ncbi_blocks:
        sids = [ncbi_to_string[n] for n in nb if n in ncbi_to_string]
        if sids:
            string_blocks.append(sids)

    # Fetch network for each block-pair (i,j) with i<=j to cover intra- and inter-block edges
    all_edge_frames = []
    for bi in range(len(string_blocks)):
        for bj in range(bi, len(string_blocks)):
            ids_pair = string_blocks[bi] + string_blocks[bj]
            params_net = {
                "species": species,
                "caller_identity": caller,
                "identifiers": "\r".join(ids_pair),
            }
            df = _post_tsv(url_net, params_net, sleep_s=sleep_s)

            # back-map to NCBI Gene IDs
            df = df[["stringId_A", "stringId_B", "score"]].copy()
            df["GeneID_i"] = df["stringId_A"].map(string_to_ncbi)
            df["GeneID_j"] = df["stringId_B"].map(string_to_ncbi)
            df["Weight"] = df["score"]

            # keep only valid, drop self-loops
            df = df.dropna(subset=["GeneID_i", "GeneID_j"])
            df = df[df["GeneID_i"] != df["GeneID_j"]]

            # canonicalize undirected edges (ensure GeneID_i < GeneID_j)
            gi = df["GeneID_i"].astype(int)
            gj = df["GeneID_j"].astype(int)
            lo = np.minimum(gi, gj).astype(str)
            hi = np.maximum(gi, gj).astype(str)
            df["GeneID_i"] = lo
            df["GeneID_j"] = hi

            # select columns
            out = df[["GeneID_i", "GeneID_j", "Weight"]].drop_duplicates()
            all_edge_frames.append(out)

    edges = pd.concat(all_edge_frames, ignore_index=True)
    edges["GeneID_i"] = edges["GeneID_i"].astype(int)
    edges["GeneID_j"] = edges["GeneID_j"].astype(int)
    return edges
