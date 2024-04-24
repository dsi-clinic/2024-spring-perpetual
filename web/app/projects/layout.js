/**
 * Layout for all project components.
 * @module app/projects/layout
 */
import Nav from "@/components/nav";

export default function ProjectsLayout({ children }) {
  return (
    <div className="flex flex-col">
      <Nav />
      <main className="flex flex-col self-center justify-center items-center py-16 px-8 w-full">{children}</main>
    </div>
  );
}
