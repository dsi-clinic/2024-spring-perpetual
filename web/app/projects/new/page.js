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
    categories: yup.array(yup.string()),
    maxNumBins: yup.number().required().min(1),
    minBinDistance: yup.number().required().min(1),
    providers: yup.array(yup.string()),
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
      categories: [
        "k12",
        "universities",
        "residential",
        "hotels",
        "offices",
        "parksrecreation",
        "groceries",
        "pharmacies",
        "libraries",
        "medical",
        "recycling",
        "airports",
        "busterminals",
        "trainstations",
      ],
      maxNumBins: 300,
      minBinDistance: 25,
      providers: ["google", "tomtom", "tripadvisor", "yelp"],
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
        <h3 className="text-slate-600 text-lg font-bold tracking-[-0.02em]">Geographic Extent</h3>
        <p className="text-foreground-500">
          Search for a municipality, census-designated place, or county by name and then edit its geographic boundary if
          needed.
        </p>
        <Controller
          name="geography"
          control={control}
          render={({ field }) => (
            <Input
              isInvalid={!!errors?.geography}
              type="search"
              placeholder="Example: Hilo, Hawaii"
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
        <h3 className="text-slate-600 text-lg font-bold tracking-[-0.02em]">Initial Outdoor Bin Placement</h3>
        <p className="text-foreground-500">
          In addition to foodware using establishments, what types of locations should receive outdoor bins?
        </p>
        <Controller
          name="categories"
          control={control}
          render={({ field }) => (
            <CheckboxGroup label="" {...field}>
              <div className="grid grid-cols-2 items-start">
                <div className="grid gap-2">
                  <p className="font-bold">Education</p>
                  <Checkbox value="preschools">Daycares and Preschools</Checkbox>
                  <Checkbox value="k12">K-12 Schools</Checkbox>
                  <Checkbox value="universities">Colleges and Universities</Checkbox>
                  <Spacer y={1} />
                  <p className="font-bold">Lodging</p>
                  <Checkbox value="residential">Large Residential Dwellings</Checkbox>
                  <Checkbox value="hotels">Large Hotels</Checkbox>
                  <Spacer y={1} />
                  <p className="font-bold">Workplaces</p>
                  <Checkbox value="offices">Office Complexes</Checkbox>
                  <Spacer y={1} />
                  <p className="font-bold">Attractions</p>
                  <Checkbox value="casinos">Casinos</Checkbox>
                  <Checkbox value="movies">Movie Theaters</Checkbox>
                  <Checkbox value="museums">Museums</Checkbox>
                  <Checkbox value="stadiums">Stadiums</Checkbox>
                  <Checkbox value="parksrecreation">Parks and Recreation</Checkbox>
                  <Checkbox value="zoos">Zoos</Checkbox>
                </div>
                <div className="grid gap-2">
                  <p className="font-bold">Shopping</p>
                  <Checkbox value="groceries">Big Box Groceries</Checkbox>
                  <Checkbox value="pharmacies">Pharmacies</Checkbox>
                  <Spacer y={1} />
                  <p className="font-bold">Transportation</p>
                  <Checkbox value="airports">Airports</Checkbox>
                  <Checkbox value="busterminals">Bus Terminals</Checkbox>
                  <Checkbox value="trainstations">Train Stations</Checkbox>
                  <Spacer y={1} />
                  <p className="font-bold">Services</p>
                  <Checkbox value="medical">Hospitals and Medical Centers</Checkbox>
                  <Checkbox value="libraries">Libraries</Checkbox>
                  <Checkbox value="postoffice">Post Offices</Checkbox>
                  <Checkbox value="mailboxes">Mail Dropboxes</Checkbox>
                  <Checkbox value="recycling">Recycling Dropoff Sites</Checkbox>
                </div>
              </div>
            </CheckboxGroup>
          )}
        />
        <p className="text-foreground-500">
          What is the maximum number of outdoor bins that should be returned for the initial map?{" "}
        </p>
        <Controller
          name="maxNumBins"
          control={control}
          render={({ field }) => (
            <Input
              isInvalid={!!errors?.maxNumBins}
              type="number"
              label=""
              errorMessage={errors?.maxNumBins?.message}
              size="lg"
              variant="underlined"
              min={1}
              {...field}
            />
          )}
        />
        <p className="text-foreground-500">
          What is the minimum distance in meters that should exist between each bin?{" "}
        </p>
        <Controller
          name="minBinDistance"
          control={control}
          render={({ field }) => (
            <Input
              isInvalid={!!errors?.minBinDistance}
              type="number"
              label=""
              errorMessage={errors?.minBinDistance?.message}
              size="lg"
              variant="underlined"
              min={1}
              {...field}
            />
          )}
        />
        <p className="text-foreground-500">
          <span className="font-bold">(Advanced)</span> What providers should be used to fetch the location data?
        </p>
        <Controller
          name="providers"
          control={control}
          render={({ field }) => (
            <CheckboxGroup label="" {...field}>
              <Checkbox value="google">Google</Checkbox>
              <Checkbox value="tomtom">TomTom</Checkbox>
              <Checkbox value="tripadvisor">TripAdvisor</Checkbox>
              <Checkbox value="yelp">Yelp</Checkbox>
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
