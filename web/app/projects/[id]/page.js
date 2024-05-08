"use client";

import { BinMap, LocaleMap } from "@/components/map";
import { PencilIcon } from "@/components/pencil_icon";
import { get } from "@/services/http";
import { Spacer, Spinner } from "@nextui-org/react";
import "maplibre-gl/dist/maplibre-gl.css";
import Link from "next/link";
import { useEffect, useState } from "react";

export default function ProjectPage({ params }) {
  const [project, setProject] = useState([]);
  const [projectLoading, setProjectLoading] = useState(true);

  const getProject = async (id) => {
    const url = `/api/projects/${id}`;
    const errMsg = `Failed to fetch project ${id}.`;
    return await get(url, errMsg);
  };

  const formatDate = (dateStr) => {
    const options = {
      month: "long",
      day: "numeric",
      year: "numeric",
    };
    return new Date(Date.parse(dateStr)).toLocaleDateString("en-US", options);
  };

  useEffect(() => {
    getProject(params.id).then((data) => {
      setProject(data);
      setProjectLoading(false);
    });
  }, []);

  if (projectLoading) {
    return <Spinner label="Loading..." color="primary" labelColor="primary" size="large" />;
  }

  return (
    <div className="flex w-full flex-col flex-grow max-w-5xl">
      <h2 className="from-black to-stone-500 text-3xl font-bold tracking-[-0.02em] pb-4">
        {project.name}{" "}
        <Link href={`/projects/${params.id}/edit`}>
          <PencilIcon size={5} fillColor="rgb(6 182 212)" />
        </Link>
      </h2>
      {project.description && (
        <>
          <p className="text-foreground-500 text-base italic">{project.description}</p>
          <Spacer y={4} />
        </>
      )}
      <p className="text-foreground-500">
        <span className="font-bold">Created On: </span>
        {formatDate(project.created_at_utc)}
      </p>
      <p className="text-foreground-500">
        <span className="font-bold">Last Updated: </span>
        {formatDate(project.last_updated_at_utc)}
      </p>
      <Spacer y={3} />
      {project.bins ? (
        <BinMap boundary={project.boundary} bins={project.bins} />
      ) : (
        <LocaleMap boundary={project.boundary} />
      )}
      <Spacer y={3} />
    </div>
  );
}
