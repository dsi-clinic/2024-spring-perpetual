"use client";

import { Spinner } from "@nextui-org/react";
import { useEffect, useState } from "react";

import ProjectForm from "../../forms";
import { projectService } from "../../services";

export default function ProjectEditPage({ params }) {
  const [project, setProject] = useState([]);
  const [projectLoading, setProjectLoading] = useState(true);

  useEffect(() => {
    projectService.get(params.id).then((data) => {
      setProject(data);
      setProjectLoading(false);
    });
  }, []);

  if (projectLoading) {
    return <Spinner label="Loading..." color="primary" labelColor="primary" size="large" />;
  }

  return (
    <div className="flex w-full flex-col flex-grow max-w-5xl">
      <h2 className="from-black to-stone-500 text-3xl font-bold tracking-[-0.02em] pb-4">Edit Project</h2>
      <ProjectForm project={project} />
    </div>
  );
}
