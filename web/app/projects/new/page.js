"use client";

import ProjectForm from "../forms";

export default function ProjectCreatePage() {
  return (
    <div className="flex w-full flex-col flex-grow max-w-5xl">
      <h2 className="from-black to-stone-500 text-3xl font-bold tracking-[-0.02em] pb-4">Create New Project</h2>
      <ProjectForm />
    </div>
  );
}
