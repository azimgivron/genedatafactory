import itertools
import re
from typing import List

import numpy as np
import pandas as pd


def compute_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all unique MIM-MIM edges per gene.

    Groups variants by GeneID, generates all pairwise MIM combinations,
    and attaches the corresponding ClinicalSignificance value.

    Args:
        df: DataFrame with columns ['GeneID', 'MIM', 'ClinicalSignificance'].

    Returns:
        DataFrame with columns ['MIM_i', 'MIM_j', 'ClinicalSignificance'].
    """
    edges = (
        df.groupby("GeneID", group_keys=False)[["MIM", "ClinicalSignificance"]]
        .apply(
            lambda g: pd.Series(
                {
                    "ClinicalSignificance": g["ClinicalSignificance"].iloc[0],
                    "edge": list(
                        itertools.combinations(sorted(set(g["MIM"].dropna())), 2)
                    ),
                }
            )
        )
        .reset_index()
    )

    edges = edges.explode("edge", ignore_index=True)
    edges[["MIM_i", "MIM_j"]] = pd.DataFrame(edges["edge"].tolist(), index=edges.index)
    edges = edges[["MIM_i", "MIM_j", "ClinicalSignificance"]]
    return edges


def extract_mim_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Extract OMIM numeric identifiers from the 'PhenotypeIDS' column.

    Args:
        df: DataFrame containing a 'PhenotypeIDS' column.

    Returns:
        DataFrame with an added 'MIM' column (list of integer OMIM IDs).
    """
    df = df.copy()
    df["MIM"] = df["PhenotypeIDS"].apply(
        lambda s: (
            [int(m) for m in re.findall(r"OMIM:(\d+)", str(s))] if pd.notna(s) else []
        )
    )
    return df


def map_cat(df: pd.DataFrame) -> pd.DataFrame:
    """Map clinical significance categories to normalized numeric scores.

    Args:
        df: DataFrame containing a 'ClinicalSignificance' column.

    Returns:
        DataFrame with 'ClinicalSignificance' replaced by numeric scores in [0, 1].
    """
    categories = [
        "Pathogenic",
        "Pathogenic/Likely pathogenic",
        "Likely pathogenic",
        "Uncertain significance",
        "Likely benign",
        "Benign",
    ]

    scores = np.cumsum(range(len(categories)), dtype=np.float64)[::-1]
    scores /= scores[0]
    score_map = {c: s for c, s in zip(categories, scores)}
    df["ClinicalSignificance"] = df["ClinicalSignificance"].map(score_map)
    df["ClinicalSignificance"] = df["ClinicalSignificance"].astype(np.float64)
    df = df[df["ClinicalSignificance"] > 0]
    return df


def read_clinvar(path: str, diseaseid: List[int]) -> pd.DataFrame:
    """Load ClinVar variant summary and build filtered MIM-MIM edge list.

    Reads ClinVar's variant_summary.txt.gz, extracts OMIM identifiers,
    filters by the provided disease IDs, converts clinical significance
    to numeric form, and constructs all pairwise MIM edges per gene.

    Args:
        path: Path to ClinVar variant_summary.txt.gz.
        diseaseid: List of OMIM integer IDs to filter by.

    Returns:
        DataFrame with columns ['MIM_i', 'MIM_j', 'ClinicalSignificance'].
    """
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        compression="gzip",
        header=None,
        names=[
            "AlleleID",
            "Type",
            "Name",
            "GeneID",
            "GeneSymbol",
            "HGNC_ID",
            "ClinicalSignificance",
            "ClinSigSimple",
            "LastEvaluated",
            "RS# (dbSNP)",
            "nsv/esv (dbVar)",
            "RCVaccession",
            "PhenotypeIDS",
            "PhenotypeList",
            "Origin",
            "OriginSimple",
            "Assembly",
            "ChromosomeAccession",
            "Chromosome",
            "Start",
            "Stop",
            "ReferenceAllele",
            "AlternateAllele",
            "Cytogenetic",
            "ReviewStatus",
            "NumberSubmitters",
            "Guidelines",
            "TestedInGTR",
            "OtherIDs",
            "SubmitterCategories",
            "VariationID",
            "PositionVCF",
            "ReferenceAlleleVCF",
            "AlternateAlleleVCF",
            "SomaticClinicalImpact",
            "SomaticClinicalImpactLastEvaluated",
            "ReviewStatusClinicalImpact",
            "Oncogenicity",
            "OncogenicityLastEvaluated",
            "ReviewStatusOncogenicity",
            "SCVsForAggregateGermlineClassification",
            "SCVsForAggregateSomaticClinicalImpact",
            "SCVsForAggregateOncogenicityClassification",
        ],
        usecols=["GeneID", "ClinicalSignificance", "PhenotypeIDS"],
        dtype={
            "GeneID": "Int64",
            "ClinicalSignificance": "string",
            "PhenotypeIDS": "string",
        },
    )

    df = map_cat(df)
    df = extract_mim_numbers(df)
    df = df.explode("MIM", ignore_index=True)
    df = df[df["MIM"].isin(diseaseid)].reset_index(drop=True)
    df = compute_edges(df)

    return df
