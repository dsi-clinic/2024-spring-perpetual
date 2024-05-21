import json
from pathlib import Path
import logging
from shapely import MultiPolygon, Polygon

logging.basicConfig(level=logging.INFO, filename="city_boundary_creation.log", filemode="w", 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
handler = logging.FileHandler("city_boundary_creation.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

### Constants ###
BOUNDARIES_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "boundaries"



class CityGeo:
    def __init__(self, cityname: str, 
                boundaries_data_path: Path = BOUNDARIES_DATA_PATH
        ):
        self.cityname = cityname
        self.cityname_geojson = self.prep_cityname_geojson(cityname)
        self.city_geojson_pathway = boundaries_data_path / f"{self.cityname_geojson}.geojson"
        self.city_geojson = self.load_boundary_geojson()
        self.geojson_type = self.city_geojson['features'][0]['geometry']['type'].lower().strip()
        self.geojson_coordinates = self.city_geojson['features'][0]['geometry']['coordinates']
        self.geo = self.city_geo()
    
    
    def prep_cityname_geojson(self, cityname: str):
        """Prepares string representing the city's name for the geojson file by 
        removing whitespace, replacing spaces with _, and converting to lowercase.
        
        Args:
            cityname (`str`): The name of the city to be prepared.

        Returns:
            cityname (`str`): The prepared city name.
        """
        cityname = cityname.strip()
        cityname = cityname.replace(" ", "_")
        cityname = cityname.lower()
        
        return cityname


    def load_boundary_geojson(self):
        """Loads the geojson file for the city.
        
        Args:
            city_geojson_pathway (`str`): The pathway to the geojson file for 
                the city.

        Returns:
            city_geojson (`dict`): The geojson file for the city.
        """
        with open(self.city_geojson_pathway, 'r') as file:
            city_geojson = json.load(file)
        
        return city_geojson
    

    def city_geo(self):
        """This method creates a shapely polygon or multipolygon object from 
        the coordinates in the geojson file for the city.

        Returns:
            polygon_s (`shapely.Polygon` or `shapely.MultiPolygon`): The 
                polygon or multipolygon object for the city.
        """
        if self.geojson_type == "polygon":
            external_poly = self.geojson_coordinates[0]
            if len(self.geojson_coordinates) == 1:
                internal_polys = None
            else:
                internal_polys = self.geojson_coordinates[1:]
            polygon_s = Polygon(external_poly, internal_polys)
            
        elif self.geojson_type == "multipolygon":
            poly_lst = []
            for polygons in self.geojson_coordinates:
                external_poly = polygons[0]
                if len(polygons) == 1:
                    internal_polys = None
                else:
                    internal_polys = polygons[1:]
                poly_lst.append(Polygon(external_poly, internal_polys))
            polygon_s = MultiPolygon(poly_lst)
            
        else:
            raise ValueError("Invalid geometry type")
        
        return polygon_s
