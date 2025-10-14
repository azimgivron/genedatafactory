from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
import yaml
from pathlib import Path
from typing import Any, Dict
from importlib.resources import files

import pandas as pd  # for type hints and saving CSVs

from genedatafactory.gene_disease.omim import read_omim
from genedatafactory.gene.go import read_go
from genedatafactory.disease.hpo import read_hpo
from genedatafactory.gene.swissprot import read_swissprot
from genedatafactory.gene.pathway import read_pathway
from genedatafactory.gene.string_net import read_string
from genedatafactory.disease.variant import read_variant


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG: Dict[str, Any] = {}
FILES = CONFIG["files"]
STRING_KWARGS = CONFIG["string_api"]
UA = CONFIG["user_agent"]


def download_with_retries(
    url: str, dest: Path, attempts: int = 4, backoff: float = 1.5, timeout: int = 60
) -> None:
    """Download a file with retries and exponential backoff.

    Args:
        url: File URL to download.
        dest: Destination file path.
        attempts: Maximum retry attempts.
        backoff: Exponential backoff base in seconds.
        timeout: Timeout per attempt in seconds.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})

    for i in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, dest.open(
                "wb"
            ) as f:
                while True:
                    chunk = r.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
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
        target_dir: Directory to store or download raw files.
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


def report(step: str, df: "pd.DataFrame", by_columns: list[str] | None = None) -> None:
    """Print summary statistics of a DataFrame.

    Args:
        step: Name of the processing step.
        df: DataFrame to summarize.
        by_columns: Columns for which to count unique values.
    """
    if by_columns is None:
        by_columns = []
    print(f"\n🧩 {step}")
    print("-" * (len(step) + 3))
    print(f"  Rows : {len(df):,}")
    for by in by_columns:
        print(f"  Unique {by} : {df[by].nunique():,}")


def save_df(df: pd.DataFrame, name: str, output_dir: Path) -> None:
    """Save a DataFrame to CSV in the output directory.

    Args:
        df: DataFrame to save.
        name: Base name of the CSV file.
        output_dir: Directory to write the file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"📦 Saved {name}.csv ({out_path.stat().st_size / 1e6:.2f} MB)")


def process_files(input_dir: Path, output_dir: Path) -> None:
    """Process all data sources and export cleaned datasets to CSV.

    Args:
        input_dir: Directory containing raw files.
        output_dir: Directory to save processed CSV files.
    """
    gd_path = input_dir / "mim2gene_medgen"
    go_path = input_dir / "go-basic.obo"
    gene2go_path = input_dir / "gene2go.gz"
    swissprot_path = input_dir / "uniprot_sprot.dat.gz"
    pathway_path = input_dir / "NCBI2Reactome_All_Levels.txt"
    variant_path = input_dir / "variant_summary.txt.gz"

    # Gene–disease relationships
    gene_disease = read_omim(str(gd_path))
    report("Gene Disease data", gene_disease, ["GeneID", "MIM number"])
    save_df(gene_disease, "gene_disease", output_dir)

    diseaseid = gene_disease["MIM number"].drop_duplicates().astype("int32").tolist()
    geneid = gene_disease["GeneID"].drop_duplicates().astype("int32").tolist()

    # HPO
    hpo = read_hpo(diseaseid)
    report("HPO data", hpo, ["MIM number"])
    save_df(hpo, "hpo", output_dir)

    # GO
    go = read_go(str(go_path), str(gene2go_path), geneid)
    report("GO data", go, ["GeneID"])
    save_df(go, "go", output_dir)

    # SwissProt
    swissprot = read_swissprot(str(swissprot_path), geneid)
    report("SWISS PROT data", swissprot, ["GeneID"])
    save_df(swissprot, "swissprot", output_dir)

    # Pathway
    pathway = read_pathway(str(pathway_path), geneid)
    report("Pathway data", pathway, ["GeneID"])
    save_df(pathway, "pathway", output_dir)

    # STRING
    string = read_string(geneid=geneid, **STRING_KWARGS)
    report("STRING data", string, ["GeneID_i"])
    save_df(string, "string", output_dir)

    # ClinVar variants
    variant = read_variant(str(variant_path), diseaseid)
    report("VARIANT data", variant, ["MIM_i"])
    save_df(variant, "variant", output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input and output directories.

    Returns:
        Parsed CLI arguments.
    """
    p = argparse.ArgumentParser(
        description="Download, process, and export biomedical reference data."
    )
    p.add_argument("-i", "--input", type=Path, required=True, help="Folder to store raw files.")
    p.add_argument(
        "-o", "--output", type=Path, required=True, help="Folder to write processed CSV files."
    )
    return p.parse_args()    

def main() -> None:
    """Entry point."""
    args = parse_args()
    global CONFIG
    config_text = (files("genedatafactory") / "config.yaml").read_text()
    CONFIG = yaml.safe_load(config_text)
    ensure_files(args.input_folder)
    process_files(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
