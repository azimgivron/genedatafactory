from __future__ import annotations

import argparse
import ssl
import sys
import time
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
import yaml

from genedatafactory.binary.disease.hpo import read_hpo
from genedatafactory.binary.disease.mondo import read_mondo
from genedatafactory.binary.gene.go import read_go
from genedatafactory.binary.gene.reactome import read_reactome
from genedatafactory.binary.gene.swissprot import read_swissprot
from genedatafactory.embeddings.text import read_generifs_basic, read_medgen_definitions
from genedatafactory.gene_disease.omim import read_omim
from genedatafactory.graph.string_net import read_string
from genedatafactory.script.utils import count, new_mapping, remap

CONFIG: Dict[str, Any] = {}
FILES = None
STRING_KWARGS = None
UA = None
NAMES = None

try:
    import certifi

    _CERTIFI = certifi.where()
except Exception:
    _CERTIFI = None


def download_with_retries(
    url: str,
    dest: Path,
    attempts: int = 4,
    backoff: float = 1.5,
    timeout: int = 60,
    proxies_from_env: bool = True,
) -> None:
    """Download a file with retry and SSL options.

    Attempts to download a file from a URL with exponential backoff on failure.
    Supports configurable SSL verification and optional proxy settings.

    Args:
        url (str): URL of the file to download.
        dest (Path): Destination path for the downloaded file.
        attempts (int, optional): Maximum retry attempts. Defaults to 4.
        backoff (float, optional): Base for exponential backoff between retries. Defaults to 1.5.
        timeout (int, optional): Timeout per attempt in seconds. Defaults to 60.
        proxies_from_env (bool, optional): Use environment proxy settings. Defaults to True.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Build SSL context
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(cafile=_CERTIFI)

    # Build opener
    handlers = []
    if proxies_from_env:
        handlers.append(urllib.request.ProxyHandler())  # respects *_PROXY env vars
    handlers.append(urllib.request.HTTPSHandler(context=ctx))
    opener = urllib.request.build_opener(*handlers)

    headers = {
        "User-Agent": UA,
    }
    req = urllib.request.Request(url, headers=headers)

    for i in range(1, attempts + 1):
        try:
            with opener.open(req, timeout=timeout) as r, dest.open("wb") as f:
                while True:
                    chunk = r.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
            return
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ssl.SSLError,
        ) as e:
            if i == attempts:
                raise
            sleep_s = backoff ** (i - 1)
            print(
                f"  Attempt {i}/{attempts} failed ({e}). Retrying in {sleep_s:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_s)


def ensure_files(target_dir: Path) -> None:
    """Ensure all required reference files exist in the input directory.

    Args:
        target_dir (Path): Directory to store or download raw files.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in FILES.items():
        dest = target_dir / fname

        if dest.exists() and dest.stat().st_size > 0:
            print(f"✅ {fname} already present ({dest.stat().st_size:,} bytes)")
            continue

        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass

        print(f"⬇️  Downloading {fname} from {url}")
        try:
            download_with_retries(url, dest)
            print(f"📦 Saved to {dest} ({dest.stat().st_size:,} bytes)")
        except Exception as e:
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            print(f"❌ Failed to download {fname}: {e}", file=sys.stderr)


def report(step: str, df: "pd.DataFrame", by_columns: List[str] | None = None) -> None:
    """Print summary statistics of a DataFrame.

    Args:
        step (str): Name of the processing step.
        df (pd.DataFrame): DataFrame to summarize.
        by_columns (List[str], optional): Columns for which to count unique values.
    """
    if by_columns is None:
        by_columns = []
    print(f"\n🧩 {step}")
    print("-" * (len(step) + 3))
    print(f"  Rows : {len(df):,}")
    for by in by_columns:
        print(
            f"  Unique {by} : {df[by].nunique():,}, ranging from {df[by].min()} to {df[by].max()}"
        )


def report_string(string: pd.DataFrame) -> None:
    """Print a summary report of a STRING protein-protein interaction dataset.

    Args:
        string (pd.DataFrame): DataFrame representing the STRING network,
            where each row corresponds to an interaction (edge) between two proteins (nodes).

    Prints:
        - Total number of edges (rows)
        - Total number of unique nodes (proteins)
    """
    step = "STRING data"
    print(f"\n🧩 {step}")
    print("-" * (len(step) + 3))
    print(f"  Edges : {len(string):,}")
    print(f"  Nodes : {len(np.unique(string.to_numpy().flatten())):,}")


def save_df(df: pd.DataFrame, name: str, output_dir: Path) -> None:
    """Save a DataFrame to CSV in the output directory.

    Args:
        df (pd.DataFrame): DataFrame to save.
        name (str): Base name of the CSV file.
        output_dir (Path): Directory to write the file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"📦 Saved {name}.csv ({out_path.stat().st_size / 1e6:.2f} MB)")


def process_files(
    input_dir: Path,
    output_dir: Path,
    filter: bool,
    sources: Set[str],
    min_genes: int = None,
) -> None:
    """Process all data sources and export cleaned datasets to CSV.

    Args:
        input_dir (str): Directory containing raw files.
        output_dir (str): Directory to save processed CSV files.
        filter (bool): Whether to keep fully described
            genes and diseases only in OMIM dataset.
        sources (Set[str]): Set of data sources to process.
        min_genes (int, optional): Minimum number of associated genes to be kept.
    """
    gd_path = input_dir / NAMES["OMIM"]
    generifs_basic_path = input_dir / NAMES["GENE_RIFS"]
    go_path = input_dir / NAMES["GO"]
    gene2go_path = input_dir / NAMES["GO_2"]
    swissprot_path = input_dir / NAMES["SWISSPROT"]
    reactome_path = input_dir / NAMES["REACTOME"]
    mondo_path = input_dir / NAMES["MONDO"]
    mgdef_path = input_dir / NAMES["MGDEF"]
    mgdef_mapping_path = input_dir / NAMES["MGDEF_MAPPING"]

    gene_key = "GeneID"
    disease_key = "MIM number"

    # OMIM relationships
    gene_disease, gene_ids = read_omim(str(gd_path))
    gene_ids = list(gene_ids)

    if min_genes is not None:
        gene_disease = gene_disease.groupby(disease_key).filter(
            lambda x: x[gene_key].nunique() > min_genes
        )
    disease_ids = gene_disease[disease_key].drop_duplicates().astype("int32").tolist()
    others = []

    if "medgen" in sources:
        medgen = read_medgen_definitions(
            str(mgdef_path), str(mgdef_mapping_path), disease_ids
        )
        others.append(medgen)
    if "gene_rifs" in sources:
        gene_rifs = read_generifs_basic(str(generifs_basic_path), gene_ids)
        others.append(gene_rifs)
    if "hpo" in sources:
        hpo = read_hpo(disease_ids)
        others.append(hpo)
    if "go" in sources:
        go = read_go(str(go_path), str(gene2go_path), gene_ids)
        others.append(go)
    if "swissprot" in sources:
        swissprot = read_swissprot(str(swissprot_path), gene_ids)
        others.append(swissprot)
    if "reactome" in sources:
        reactome = read_reactome(str(reactome_path), gene_ids)
        others.append(reactome)
    if "mondo" in sources:
        mondo = read_mondo(str(mondo_path), disease_ids)
        others.append(mondo)
    if "string" in sources:
        string = read_string(gene_ids=gene_ids, **STRING_KWARGS)
        string_genes = pd.DataFrame(
            np.unique(string.to_numpy().flatten()).T, columns=[gene_key]
        )
        others.append(string_genes)

    if filter:
        print(
            "\n⚠️  Filtering: Keep fully characterized genes and diseases (no missing feature) ⚠️"
        )
        genes, diseases = count(set(gene_ids), set(disease_ids), others=others)
        mask = (gene_disease[gene_key].isin(genes)) & (
            gene_disease[disease_key].isin(diseases)
        )
        gene_disease = gene_disease[mask]
        diseases = set(gene_disease[disease_key])

        gene_mapping = {
            np.int32(old_id): np.int32(new_id) for new_id, old_id in enumerate(genes)
        }
        disease_mapping = {
            np.int32(old_id): np.int32(new_id) for new_id, old_id in enumerate(diseases)
        }
        gene_disease = new_mapping(gene_disease, gene_key, gene_mapping)
        gene_disease = new_mapping(gene_disease, disease_key, disease_mapping)
        gene_disease = gene_disease.rename(columns={disease_key: "DiseaseID"})

        if "gene_rifs" in sources:
            gene_rifs = gene_rifs[gene_rifs[gene_key].isin(genes)]
            gene_rifs = new_mapping(gene_rifs, gene_key, gene_mapping)
        if "go" in sources:
            go = go[go[gene_key].isin(genes)]
            go = new_mapping(go, gene_key, gene_mapping)
            go = remap(go, 1)
        if "swissprot" in sources:
            swissprot = swissprot[swissprot[gene_key].isin(genes)]
            swissprot = new_mapping(swissprot, gene_key, gene_mapping)
            swissprot = remap(swissprot, 1)
        if "reactome" in sources:
            reactome = reactome[reactome[gene_key].isin(genes)]
            reactome = new_mapping(reactome, gene_key, gene_mapping)
            reactome = remap(reactome, 1)
        if "medgen" in sources:
            medgen = medgen[medgen[disease_key].isin(diseases)]
            medgen = new_mapping(medgen, disease_key, disease_mapping)
            medgen = medgen.rename(columns={disease_key: "DiseaseID"})
        if "hpo" in sources:
            hpo = hpo[hpo[disease_key].isin(diseases)]
            hpo = new_mapping(hpo, disease_key, disease_mapping)
            hpo = remap(hpo, 1)
            hpo = hpo.rename(columns={disease_key: "DiseaseID"})
        if "mondo" in sources:
            mondo = mondo[mondo[disease_key].isin(diseases)]
            mondo = new_mapping(mondo, disease_key, disease_mapping)
            mondo = remap(mondo, 1)
            mondo = mondo.rename(columns={disease_key: "DiseaseID"})
        if "string" in sources:
            string = string[string[f"{gene_key}_i"].isin(genes)]
            string = string[string[f"{gene_key}_j"].isin(genes)]
            string = new_mapping(string, f"{gene_key}_i", gene_mapping)
            string = new_mapping(string, f"{gene_key}_j", gene_mapping)
        disease_key = "DiseaseID"

    print(f"Total number of genes: {len(genes)}")
    report("Gene Disease data", gene_disease, [gene_key, disease_key])
    save_df(gene_disease, "gene_disease", output_dir)

    if "medgen" in sources:
        report("Disease MEDGEN text data", medgen, [disease_key])
        save_df(medgen, "medgen", output_dir)
    if "gene_rifs" in sources:
        report("Gene RIFS data", gene_rifs, [gene_key])
        save_df(gene_rifs, "gene_rifs", output_dir)
    if "hpo" in sources:
        report("HPO data", hpo, [disease_key, hpo.columns[1]])
        save_df(hpo, "hpo", output_dir)
    if "go" in sources:
        report("GO data", go, [gene_key, go.columns[1]])
        save_df(go, "go", output_dir)
    if "swissprot" in sources:
        report("SWISS PROT data", swissprot, [gene_key, swissprot.columns[1]])
        save_df(swissprot, "swissprot", output_dir)
    if "reactome" in sources:
        report("Reactome data", reactome, [gene_key, reactome.columns[1]])
        save_df(reactome, "reactome", output_dir)
    if "mondo" in sources:
        report("Mondo data", mondo, [disease_key, mondo.columns[1]])
        save_df(mondo, "mondo", output_dir)
    if "string" in sources:
        report_string(string)
        save_df(string, "string", output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input and output directories.

    Returns:
        Parsed CLI arguments.
    """
    p = argparse.ArgumentParser(
        description=(
            "🧬 Download, curate, and export biomedical reference data,\n "
            "an automated pipeline for standardized gene-disease resources."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "-i", "--input", type=Path, required=True, help="Folder to store raw files."
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Folder to write processed CSV files.",
    )
    p.add_argument(
        "-f",
        "--filter",
        action="store_true",
        help="Filter the dataset to retain only fully\ncharacterized genes and diseases in OMIM.",
    )
    p.add_argument(
        "-s",
        "--sources",
        nargs="*",
        choices=[
            "go",
            "hpo",
            "mondo",
            "swissprot",
            "reactome",
            "medgen",
            "gene_rifs",
            "string",
        ],
        help="Optional list of data sources to process (e.g., go hpo).",
    )
    p.add_argument(
        "-g",
        "--min-genes",
        type=int,
        required=False,
        default=None,
        help="Optional minimum number of genes for a disease to be kept (default: %(default)s).",
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    global CONFIG, FILES, STRING_KWARGS, UA, NAMES
    config_text = (files("genedatafactory") / "config.yaml").read_text()
    CONFIG = yaml.safe_load(config_text)
    FILES = CONFIG["url"]
    NAMES = CONFIG["files"]
    STRING_KWARGS = CONFIG["string_api"]
    UA = CONFIG["user_agent"]
    ensure_files(args.input)
    sources = (
        set(args.sources)
        if args.sources
        else {
            "omim",
            "go",
            "hpo",
            "mondo",
            "swissprot",
            "reactome",
            "medgen",
            "gene_rifs",
            "string",
        }
    )
    process_files(
        args.input, args.output, args.filter, sources, min_genes=args.min_genes
    )


if __name__ == "__main__":
    main()
