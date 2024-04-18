"use client";

import { SearchIcon } from "@/components/search_icon";
import { yupResolver } from "@hookform/resolvers/yup";
import { Button, Checkbox, CheckboxGroup, Input, Spacer, Textarea } from "@nextui-org/react";
import "maplibre-gl/dist/maplibre-gl.css";
import { Controller, useForm } from "react-hook-form";
import Map from "react-map-gl/maplibre";
import * as yup from "yup";

export default function ProjectFormPage() {
  const projectSchema = yup.object({
    name: yup.string().required("Name is a required field.").min(1).max(255),
    description: yup.string().max(1000),
    geography: yup.string().required("Must enter at least one geography."),
  });

  const {
    handleSubmit,
    control,
    formState: { errors },
  } = useForm({
    resolver: yupResolver(projectSchema),
    mode: "onSubmit",
    defaultValues: {
      name: "",
      description: "",
      geography: "",
    },
  });

  const onSubmit = (data) => {
    alert(JSON.stringify(data));
  };

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

  return (
    <div className="flex w-full flex-col flex-grow max-w-5xl">
      <h2 className="from-black to-stone-500 text-3xl font-bold tracking-[-0.02em] pb-4">Create New Project</h2>
      <form className="flex flex-col gap-4" onSubmit={handleSubmit(onSubmit)}>
        <h3 className="text-slate-600 text-lg font-bold tracking-[-0.02em]">Background</h3>
        <Controller
          name="name"
          control={control}
          render={({ field }) => (
            <Input
              isInvalid={!!errors?.name}
              type="text"
              label="Name"
              placeholder="Enter the project name"
              errorMessage={errors?.name?.message}
              size="lg"
              variant="underlined"
              {...field}
            />
          )}
        />
        <Controller
          name="description"
          control={control}
          render={({ field }) => (
            <Textarea
              isInvalid={!!errors?.description}
              label="Description"
              placeholder="Enter the project description"
              errorMessage={errors?.description?.message}
              size="lg"
              variant="underlined"
              {...field}
            />
          )}
        />
        <Controller
          name="geography"
          control={control}
          render={({ field }) => (
            <Input
              isInvalid={!!errors?.geography}
              type="search"
              label="Geographic Extent"
              placeholder="Search for a municipality, county, or other geography and edit boundary as necessary"
              errorMessage={errors?.geography?.message}
              size="lg"
              variant="underlined"
              startContent={<SearchIcon size={18} />}
              {...field}
            />
          )}
        />
        <Spacer y={1} />
        <Map
          initialViewState={{
            longitude: -95.7129,
            latitude: 37.0902,
            zoom: 3,
          }}
          style={{ width: "100%", height: 400 }}
          mapStyle={style}
        />
        <h3 className="text-slate-600 text-lg font-bold tracking-[-0.02em]">Initial Bin Placements</h3>
        <Controller
          name="categories"
          control={control}
          render={({ field }) => (
            <CheckboxGroup label="Select Categories" defaultValue={["hotel", "residential"]} {...field}>
              <div className="grid gap-2">
                <p className="italic">Education</p>
                <Checkbox value="k12">Preschools</Checkbox>
                <Checkbox value="k12">K-12 Schools</Checkbox>
                <Checkbox value="universities">Colleges and Universities</Checkbox>
                <Checkbox value="residential">Large Residential Dwellings</Checkbox>
                <Checkbox value="hotel">Large Hotels</Checkbox>
                <Checkbox value="">San Francisco</Checkbox>
              </div>
            </CheckboxGroup>
          )}
        />
        <Spacer y={2} />
        <Button className="self-center" color="primary" type="submit" size="lg">
          Submit
        </Button>
      </form>
    </div>
  );
}
