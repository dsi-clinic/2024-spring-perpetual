import bbox from "@turf/bbox";
import { useCallback, useRef, useState } from "react";
import Map, { Layer, Source, useMap } from "react-map-gl/maplibre";

export default function LocaleMap({ boundary }) {
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
    console.log(boundary);
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
