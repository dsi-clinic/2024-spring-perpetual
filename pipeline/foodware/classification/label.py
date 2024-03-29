"""Labels a set of standardized points of interest as indoor or outdoor bins.
"""

# Standard library imports
from typing import Dict, List, Optional

# Third-party imports
import pandas as pd


def filter(
    clean_places: List[Dict], crosswalk: Dict, random_seed: Optional[int] = 12345
) -> List[Dict]:
    """Filters a list of cleaned places to return only indoor and/or outdoor bins."""
    # Load cleaned places into DataFrame
    df = pd.DataFrame(clean_places)

    # Invert category crosswalk
    crosswalk_inv = {}
    for cat, map in crosswalk.items():
        for source, aliases in map.items():
            crosswalk_inv[source] = crosswalk_inv.get(source, {})
            for alias in aliases:
                crosswalk_inv[source][alias] = cat

    # Define local function to map place aliases to standard category names
    def map_categories(row: pd.Series) -> List[str]:
        standard_aliases = []
        for alias in row["aliases"].split("|"):
            try:
                standard_aliases.append(crosswalk_inv[row["source"]][alias])
            except KeyError:
                pass
        return list(set(standard_aliases))

    # Map to standard categories
    df["mapped_alias"] = df.apply(map_categories, axis=1)

    # Define local function to filter by multiple standard categories
    def get_by_category(aliases: List[str], whitelisted_aliases: List[str]) -> bool:
        for alias in aliases:
            if alias in whitelisted_aliases:
                return True
        return False

    # Identify FUEs
    fues = df[df["mapped_alias"].apply(lambda lst: "foodwareUsingEstablishment" in lst)]

    # Randomly select 25 percent of FUEs as bin locations
    fues_outdoor = fues.sample(frac=0.25, random_state=random_seed)

    # Identify remaining categories
    cats = [
        "education",
        "entertainment",
        "lodging",
        "medical",
        "publicServices",
        "shopping",
        "tranportation",
    ]
    other_outdoor = df[df["mapped_alias"].apply(lambda lst: get_by_category(lst, cats))]

    # Concatenate DataFrames and drop any duplicates
    all_outdoor = pd.concat([fues_outdoor, other_outdoor])
    all_outdoor = all_outdoor.drop_duplicates(subset="id")

    # Add bin type column
    all_outdoor["bin_type"] = "outdoor"

    return all_outdoor.to_dict(orient="records")
