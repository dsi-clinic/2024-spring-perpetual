import "../css/map_overrides.css";
import Pin from "@/components/pin";
import { Chip, Spacer } from "@nextui-org/react";
import bbox from "@turf/bbox";
import { useCallback, useMemo, useRef, useState } from "react";
import Map, { Layer, Marker, Popup, Source, useMap } from "react-map-gl/maplibre";

function LocaleMap({ boundary }) {
  const style = {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "&copy; OpenStreetMap Contributors",
        maxzoom: 19,
      },
    },
    layers: [
      {
        id: "osm",
        type: "raster",
        source: "osm", // This must match the source key above
      },
    ],
  };

  const mapRef = useRef();

  if (boundary && mapRef.current) {
    const [minLng, minLat, maxLng, maxLat] = bbox(boundary);
    mapRef.current.fitBounds(
      [
        [minLng, minLat],
        [maxLng, maxLat],
      ],
      { padding: 40, duration: 3000 },
    );
  }

  return (
    <Map
      ref={mapRef}
      initialViewState={{
        longitude: -95.7129,
        latitude: 37.0902,
        zoom: 3,
      }}
      style={{ width: "100%", height: 400 }}
      mapStyle={style}
    >
      {boundary && (
        <Source id="project-boundary" type="geojson" data={boundary}>
          <Layer
            id="project-layer"
            type="fill"
            paint={{
              "fill-color": "#4E3FC8",
              "fill-opacity": 0.3,
            }}
          />
        </Source>
      )}
    </Map>
  );
}

function BinMap({ boundary, bins }) {
  const [popupInfo, setPopupInfo] = useState(null);
  const [minLng, minLat, maxLng, maxLat] = bbox(boundary);

  const style = {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "&copy; OpenStreetMap Contributors",
        maxzoom: 19,
      },
    },
    layers: [
      {
        id: "osm",
        type: "raster",
        source: "osm",
      },
    ],
  };

  const pins = useMemo(
    () =>
      bins
        ? bins.map((bin, index) => {
            return (
              <Marker
                key={`marker-${index}`}
                longitude={bin.coords.coordinates[0]}
                latitude={bin.coords.coordinates[1]}
                anchor="bottom"
                onClick={(e) => {
                  e.originalEvent.stopPropagation();
                  setPopupInfo(bin);
                }}
              >
                <Pin fillColor={"red"} />
              </Marker>
            );
          })
        : [],
    [],
  );

  return (
    <Map
      initialViewState={{
        longitude: (minLng + maxLng) / 2,
        latitude: (minLat + maxLat) / 2,
        zoom: 12,
      }}
      style={{ width: "100%", height: 600 }}
      mapStyle={style}
    >
      {boundary && (
        <Source id="project-boundary" type="geojson" data={boundary}>
          <Layer
            id="project-layer"
            type="fill"
            paint={{
              "fill-color": null,
              "fill-opacity": 0.3,
            }}
          />
        </Source>
      )}

      {pins}
      {popupInfo && (
        <Popup
          anchor="top"
          longitude={Number(popupInfo.coords.coordinates[0])}
          latitude={Number(popupInfo.coords.coordinates[1])}
          onClose={() => setPopupInfo(null)}
        >
          <div>
            <Chip color="primary" size="sm" className="mb-2 mr-1">
              {popupInfo.parent_category_name}
            </Chip>
            <Chip color="secondary" size="sm" className="mb-2">
              {popupInfo.classification}
            </Chip>
            <h4 className="font-bold text-sm">{popupInfo.name.toUpperCase()}</h4>
            <div className="text-sm">{popupInfo.formatted_address.split(", ").slice(0, -1).join(", ")}</div>
          </div>
          <Spacer y={3} />
          <div>SOURCE: {popupInfo.provider_name.toUpperCase()}</div>
          <div>LAST UPDATED: {new Date(popupInfo.created_at_utc).toLocaleDateString()}</div>
        </Popup>
      )}
    </Map>
  );
}

export { LocaleMap, BinMap };
