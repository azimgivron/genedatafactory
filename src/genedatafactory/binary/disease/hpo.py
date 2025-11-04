from typing import List

import pandas as pd
from pyhpo import Ontology
from pyhpo.annotations import Omim


def read_hpo(omim_id: List[int]) -> pd.DataFrame:
    """Retrieve HPO term information for a list of OMIM disease IDs.

    Args:
        omim_id (List[int]):
            List of OMIM disease identifiers (e.g., [104300, 114480]).

    Returns:
        pd.DataFrame:
            A DataFrame with the following columns:
            - MIM number: The OMIM disease identifier.
            - TermID: The HPO term identifier (e.g., "HP:0001250").
            - Value: The information content (IC) value of the term with
              respect to the OMIM annotation set.
    """
    # Initialize ontology (downloads files if not cached)
    Ontology()
    data = []
    for oid in omim_id:
        try:
            disease = Omim.get(oid, set())
            data.extend([[oid, term_id] for term_id in disease.hpo])
        except KeyError:
            continue
    df = pd.DataFrame(data, columns=["MIM number", "TermID"])
    return df.drop_duplicates()