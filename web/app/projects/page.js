"use client";

/**
 * Page permitting authenticated users to browse projects.
 * @module app/page
 */
import PlusCircleIcon from "@/components/plus_circle";
import { useUser } from "@auth0/nextjs-auth0/client";
import { Spacer, Table, TableBody, TableCell, TableColumn, TableHeader, TableRow } from "@nextui-org/react";
import { Button } from "@nextui-org/react";
import Link from "next/link";

export default function ProjectsPage() {
  const { user, error, isLoading } = useUser();

  return (
    user && (
      <div className="justify-center items-center max-w-5xl">
        <div className="z-10 w-full px-5 xl:px-0">
          <h1
            className="animate-fade-up bg-gradient-to-br from-black to-stone-500 bg-clip-text text-center font-display font-bold tracking-[-0.02em] text-transparent drop-shadow-sm [text-wrap:balance] text-5xl leading-[5rem] p-1"
            style={{ animationDelay: "0.15s", animationFillMode: "forwards" }}
          >
            Welcome, {user.given_name}!
          </h1>
        </div>
        <div className="py-4">
          Vivamus dapibus aliquam magna quis rhoncus. Vivamus viverra risus dolor, eget vehicula diam mattis eu. Donec
          eget interdum justo. Nam ut interdum elit. Phasellus porttitor congue erat, non cursus mauris varius cursus.
          Pellentesque at nulla eu orci tempor vestibulum. Nullam blandit nisl convallis nisl auctor vehicula. Quisque
          tempor magna id nibh bibendum porta. Nam tristique leo risus, efficitur sagittis diam lobortis quis. Etiam
          pretium est sed viverra placerat.
        </div>
        <Spacer y={8} />
        <div className="flex flex-row space-x-4">
          <h2 className="from-black to-stone-500 text-3xl font-bold tracking-[-0.02em] pb-4">Design Projects</h2>
          <Link href="/projects/new">
            <Button isIconOnly radius="full" color="primary" variant="solid" size="sm">
              <PlusCircleIcon fillColor="none" strokeColor="white" strokeWidth={3} />
            </Button>
          </Link>
        </div>
        <Table>
          <TableHeader>
            <TableColumn>NAME</TableColumn>
            <TableColumn>TOTAL BINS</TableColumn>
            <TableColumn>INDOOR BINS</TableColumn>
            <TableColumn>OUTDOOR BINS</TableColumn>
            <TableColumn>CREATED ON</TableColumn>
            <TableColumn>LAST UPDATED ON</TableColumn>
          </TableHeader>
          <TableBody emptyContent={"No rows to display."}>{[]}</TableBody>
        </Table>
      </div>
    )
  );
}
