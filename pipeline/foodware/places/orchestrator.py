"""Classes used to orchestrate the fetch and transformation of place data.
"""

# Standard library imports
from logging import Logger
from typing import Dict, List, Union

# Third-party imports
import pandas as pd
import geopandas as gpd
from fuzzywuzzy import fuzz
from shapely import MultiPolygon, Polygon

# Application imports
from foodware.places.factory import IPlacesProvider, IPlacesProviderFactory


class PlaceOrchestrator:
    """Orchestrates the retrieval of relevant POI using available providers."""

    def get_top_businesses(
        self,
        infogroup_gdf: gpd.GeoDataFrame,
        locale_name: str,
        locale_boundary: Union[Polygon, MultiPolygon],
        infogroup_year: int,
        places_provider: IPlacesProvider,
        addr_threshold: int = 100,
    ) -> List[Dict]:
        """Extracts the top quartile of businesses by sales volume
        from an Infogroup (Data Axle) dataset. Then matches the
        resulting records with places retrieved from a more up-to-date
        `IPlacesProvider` like Google Maps Platform on similar name
        and address values to filter out businesses that have since closed.

        The library `fuzzywuzzy` is used to produce similarity scores between
        pairs of strings. String edit distance is calculated using the
        shortest string of length N against all N-length substrings in
        the larger string (i.e., a partial ratio).

        Args:
            infogroup_gdf (`gpd.GeoDataFrame`): The business records.

            locale_name (`str`): The name of the geography used to filter
                the business records. Used for informational display.

            locale_boundary (`shapely.Polygon`|`shapely.MultiPolygon`): The
                geographic boundary used to filter the business records.

            infogroup_year (`int`): The publication year of the InfoGroup dataset.
                Used for informational display.

            places_provider (`IPlacesProvider`): The provider used to fetch
                current points of interest.

            addr_threshold (`int`): The similarity score at which a
                provider POI street address "matches" a business street
                address. Ranges from one to one hundred inclusive
                (i.e., `[1, 100]`). Defaults to 100.

        Returns:
            (`list` of `dict`): The bin locations.
        """
        # Retain only businesses within boundary
        gdf = gdf[infogroup_gdf.intersects(locale_boundary)]

        # Determine cutoff for top 25 percent of sales volume
        sales_vol_threshold = gdf["SALES VOLUME (9) - LOCATION"].describe()["75%"]

        # Filter to top performing businesses and add rank as new column
        top_biz_gdf = gdf.query(
            "`SALES VOLUME (9) - LOCATION` > @sales_vol_threshold"
        ).sort_values(by="SALES VOLUME (9) - LOCATION", ascending=False)
        top_biz_gdf["RANK"] = range(1, len(top_biz_gdf) + 1)

        # Call provider to fetch latest state of businesses
        top_biz = []
        for _, row in top_biz_gdf.iterrows():

            # Skip business if company name was not recorded
            if not row["COMPANY"]:
                continue

            # Send search request with name and address
            rank = row["RANK"]
            sales = row["SALES VOLUME (9) - LOCATION"]
            company = row["COMPANY"]
            street_address = row["ADDRESS LINE 1"]
            city = row["CITY"]
            state = row["STATE"]
            query = ", ".join([company, street_address, city, state])
            results = places_provider.run_text_search(query, locale_boundary)

            # Process results
            try:
                # Parse best match (first result) if exists; otherwise continue
                best_match = results.clean[0]
                best_match_name = best_match.name.upper() if best_match.name else ""
                best_match_address = best_match.address.split(",")[0].upper()

                # Compute similarity scores between business and match fields
                addr_score = fuzz.partial_ratio(street_address, best_match_address)
                name_score = fuzz.partial_ratio(company, best_match_name)

                # Skip row if address match below threshold
                if addr_score < addr_threshold:
                    continue

                # Otherwise, add notes to match and append to list of top businesss
                best_match.notes = (
                    f"(Source: InfoGroup) In {infogroup_year}, this location "
                    f"housed the restaurant {company}, which had a business "
                    f"sales volume of {sales * 1000:,}. Among restaurants in "
                    f"the locale of {locale_name}, it ranked {rank} of "
                    f"{len(gdf)} in sales volume. It is expected with {name_score} "
                    "percent confidence that the same restaurant exists at this "
                    "location. Please confirm for accuracy."
                )
                top_biz.append(best_match)

            except IndexError:
                continue

        return top_biz
