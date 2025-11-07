import time
from io import StringIO
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd
import requests


def _chunks(lst: List[int], size: int) -> Iterable[List[int]]:
    """Yield successive fixed-size chunks from a list.

    Args:
        lst (List[int]): Input list to split.
        size (int): Number of elements per chunk.

    Yields:
        Iterable[List[int]]: Consecutive sublists of given size.
    """
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _post_tsv(url: str, data: Dict, sleep_s: float = 0.3) -> pd.DataFrame:
    """Send a POST request expecting TSV data with a header.

    Args:
        url (str): Target URL.
        data (Dict): Form data to send.
        sleep_s (float, optional): Pause duration after request (default 0.3).

    Returns:
        pd.DataFrame: Parsed TSV data, or empty DataFrame if no content.
    """
    r = requests.post(url, data=data, timeout=120)
    r.raise_for_status()
    txt = r.text.strip()
    if not txt:
        time.sleep(sleep_s)
        return pd.DataFrame()
    df = pd.read_csv(StringIO(txt), sep="\t")
    time.sleep(sleep_s)
    return df


def get_mappings(
    gene_ids: List[int],
    map_batch: int,
    species: int,
    caller: str,
    url_map: str,
    sleep_s: float,
) -> pd.DataFrame:
    """Maps NCBI Gene IDs to STRING protein IDs.

    Args:
        gene_ids (List[int]): NCBI Gene IDs to include.
        map_batch (int): Mapping batch size.
        species (int): NCBI taxon ID.
        caller (str): Identifier for API logging.
        url_map (str): STRING API URL for ID mapping.
        sleep_s (float): Delay between requests.

    Returns:
        pd.DataFrame: Mapping of NCBI Gene IDs to STRING protein IDs.
    """
    mappings = []
    for batch in _chunks(gene_ids, map_batch):
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
    return mapping


def get_edges(
    mapped_ncbi: Set[int],
    net_batch: int,
    string_to_ncbi: Dict[str, int],
    species: int,
    caller: str,
    url_net: str,
    sleep_s: float,
) -> pd.DataFrame:
    """Get and construct the PPI network.

    Args:
        mapped_ncbi (Set[int]): Set of mapped NCBI Gene IDs.
        net_batch (int): Network batch size.
        string_to_ncbi (Dict[str, int]): Mapping of STRING protein
            IDs to NCBI Gene IDs.
        species (int): NCBI taxon ID.
        caller (str): Identifier for API logging.
        url_net (str): STRING API URL for network retrieval.
        sleep_s (float): Delay between requests.

    Returns:
        pd.DataFrame: Undirected edge list with columns:
            - GeneID_i (int)
            - GeneID_j (int)
            - Weight (float)
    """
    ncbi_sorted = sorted(mapped_ncbi, key=lambda x: int(x))
    ncbi_blocks = list(_chunks(ncbi_sorted, net_batch))
    string_blocks = []
    ncbi_to_string = {v: k for k, v in string_to_ncbi.items()}
    for nb in ncbi_blocks:
        sids = [ncbi_to_string[n] for n in nb if n in ncbi_to_string]
        if sids:
            string_blocks.append(sids)

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
            df = df[["stringId_A", "stringId_B", "score"]].copy()
            df["GeneID_i"] = df["stringId_A"].map(string_to_ncbi)
            df["GeneID_j"] = df["stringId_B"].map(string_to_ncbi)
            df["Weight"] = df["score"]
            df = df.dropna(subset=["GeneID_i", "GeneID_j"])
            df = df[df["GeneID_i"] != df["GeneID_j"]]

            gi = df["GeneID_i"].astype(int)
            gj = df["GeneID_j"].astype(int)
            lo = np.minimum(gi, gj).astype(str)
            hi = np.maximum(gi, gj).astype(str)
            df["GeneID_i"] = lo
            df["GeneID_j"] = hi

            out = df[["GeneID_i", "GeneID_j", "Weight"]].drop_duplicates()
            all_edge_frames.append(out)

    edges = pd.concat(all_edge_frames, ignore_index=True)
    edges["GeneID_i"] = edges["GeneID_i"].astype(int)
    edges["GeneID_j"] = edges["GeneID_j"].astype(int)
    return edges


def ensure_symmetric_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure for every (i, j, w) there is a (j, i, w) with identical weight.

    If both directions exist with different weights, we resolve to a single
    undirected weight per pair (default: mean), then duplicate to both directions.

    Args:
        df (pd.DataFrame): Graph adjacency matrix.

    Returns:
        pd.DataFrame: Symmetric graph adjacency matrix.
    """
    tmp = df.copy()
    u = tmp[["GeneID_i", "GeneID_j"]].min(axis=1)
    v = tmp[["GeneID_i", "GeneID_j"]].max(axis=1)
    tmp["u"], tmp["v"] = u, v
    agg = tmp.groupby(["u", "v"], as_index=False)["Weight"].mean()
    fwd = agg.rename(columns={"u": "GeneID_i", "v": "GeneID_j"})
    bkd = agg.rename(columns={"u": "GeneID_j", "v": "GeneID_i"})
    sym = pd.concat([fwd, bkd], ignore_index=True)
    sym = sym.astype({"GeneID_i": int, "GeneID_j": int, "Weight": float})
    sym = sym.sort_values(["GeneID_i", "GeneID_j"]).reset_index(drop=True)
    return sym


def read_string(
    url_map: str,
    url_net: str,
    gene_ids: List[int],
    caller: str = "user",
    species: int = 9606,
    map_batch: int = 1000,
    net_batch: int = 1000,
    sleep_s: float = 0.3,
) -> pd.DataFrame:
    """Construct a weighted gene-gene interaction network from STRING.

    Maps NCBI Gene IDs to STRING protein IDs, retrieves interactions in batches,
    and produces an undirected network with confidence scores.

    Args:
        url_map (str): STRING API URL for ID mapping.
        url_net (str): STRING API URL for network retrieval.
        gene_ids (List[int]): NCBI Gene IDs to include.
        caller (str, optional): Identifier for API logging. Defaults to "user".
        species (int, optional): NCBI taxon ID. Defaults to 9606 (human).
        map_batch (int, optional): Mapping batch size. Defaults to 1000.
        net_batch (int, optional): Network batch size. Defaults to 1000.
        sleep_s (float, optional): Delay between requests. Defaults to 0.3.

    Returns:
        pd.DataFrame: Undirected edge list with columns:
            - GeneID_i (int)
            - GeneID_j (int)
            - Weight (float)
    """
    mapping = get_mappings(gene_ids, map_batch, species, caller, url_map, sleep_s)

    string_to_ncbi = dict(zip(mapping["stringId"], mapping["queryItem"]))
    mapped_ncbi = set(mapping["queryItem"])

    edges = get_edges(
        mapped_ncbi, net_batch, string_to_ncbi, species, caller, url_net, sleep_s
    )
    # Enforce symmetric representation
    edges = ensure_symmetric_edges(edges)
    return edges
