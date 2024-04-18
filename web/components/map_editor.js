"use client";

import { useCallback, useState } from "react";
import Map from "react-map-gl";

import ControlPanel from "./control-panel";
import DrawControl from "./draw-control";

const TOKEN = ""; // Set your mapbox token here

export default function App() {
  const [features, setFeatures] = useState({});

  const onUpdate = useCallback((e) => {
    setFeatures((currFeatures) => {
      const newFeatures = { ...currFeatures };
      for (const f of e.features) {
        newFeatures[f.id] = f;
      }
      return newFeatures;
    });
  }, []);

  const onDelete = useCallback((e) => {
    setFeatures((currFeatures) => {
      const newFeatures = { ...currFeatures };
      for (const f of e.features) {
        delete newFeatures[f.id];
      }
      return newFeatures;
    });
  }, []);

  return (
    <>
      <Map
        initialViewState={{
          longitude: -91.874,
          latitude: 42.76,
          zoom: 12,
        }}
        mapStyle="mapbox://styles/mapbox/satellite-v9"
        mapboxAccessToken={TOKEN}
      >
        <DrawControl
          position="top-left"
          displayControlsDefault={false}
          controls={{
            polygon: true,
            trash: true,
          }}
          defaultMode="draw_polygon"
          onCreate={onUpdate}
          onUpdate={onUpdate}
          onDelete={onDelete}
        />
      </Map>
      <ControlPanel polygons={Object.values(features)} />
    </>
  );
}

export function renderToDom(container) {
  createRoot(container).render(<App />);
}
