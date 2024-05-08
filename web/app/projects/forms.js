"use client";

import LocaleMap from "@/components/map";
import { SearchIcon } from "@/components/search_icon";
import { yupResolver } from "@hookform/resolvers/yup";
import { Autocomplete, AutocompleteItem, Button, Input, Spacer, Textarea } from "@nextui-org/react";
import debounce from "lodash/debounce";
import "maplibre-gl/dist/maplibre-gl.css";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import * as yup from "yup";

import { localeService, projectService } from "./services";

export default function ProjectForm({ project }) {
  const isEdit = !!project;
  const router = useRouter();
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedLocale, setSelectedLocale] = useState(isEdit ? project.locale_id : "");
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [boundary, setBoundary] = useState(isEdit ? project.boundary : "");

  useEffect(() => {
    setIsLoading(true);
    localeService.list(searchTerm).then((data) => {
      setSearchResults(data);
      setIsLoading(false);
    });
  }, [searchTerm]);

  useEffect(() => {
    if (selectedLocale) {
      localeService.get(selectedLocale).then((data) => {
        setBoundary(data);
      });
    }
  }, [selectedLocale]);

  const onSelectionChange = (id, controllerFunc) => {
    setSelectedLocale(id);
    controllerFunc(id);
  };

  const onInputChange = useCallback(
    debounce((value) => {
      setSearchTerm(value);
    }, 100),
    [],
  );

  const projectSchema = yup.object({
    name: yup.string().required("Name is a required field.").min(1).max(255),
    description: yup.string().max(1000),
    geography: yup.number().required("Must enter a geography."),
  });

  const {
    handleSubmit,
    control,
    formState: { errors },
  } = useForm({
    resolver: yupResolver(projectSchema),
    mode: "onSubmit",
    defaultValues: isEdit
      ? {
          name: project["name"],
          description: project["description"],
          geography: project["locale_id"],
        }
      : {
          name: "",
          description: "",
          geography: null,
        },
  });

  const onSubmit = (formValues) => {
    if (isEdit) {
      projectService.update(project.id, formValues).then((res) => {
        if (!res?.error) {
          router.push(`/projects/${project.id}`);
        }
      });
    } else {
      projectService.create(formValues).then((res) => {
        if (!res?.error) {
          router.push("/projects");
        }
      });
    }
  };

  return (
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
      <p className="text-foreground-500">Search for the city or county to contain the foodware system.</p>
      <Controller
        name="geography"
        control={control}
        render={({ field: { onChange } }) => (
          <Autocomplete
            variant="underlined"
            isInvalid={!!errors?.geography}
            errorMessage={errors?.geography?.message}
            allowsCustomValue={false}
            startContent={<SearchIcon size={18} />}
            items={searchResults}
            defaultInputValue={isEdit ? project.locale_name : ""}
            selectorIcon={null}
            isLoading={isLoading}
            onInputChange={onInputChange}
            onSelectionChange={(e) => onSelectionChange(e, onChange)}
            size="lg"
          >
            {(item) => (
              <AutocompleteItem key={item.id} value={item.id}>
                {item.name}
              </AutocompleteItem>
            )}
          </Autocomplete>
        )}
      />
      <Spacer y={1} />
      <LocaleMap boundary={boundary} />
      <Spacer y={2} />
      <Button className="self-center" color="primary" type="submit" size="lg">
        Submit
      </Button>
    </form>
  );
}
