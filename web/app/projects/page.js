"use client";

/**
 * Page permitting authenticated users to browse projects.
 * @module app/page
 */
import { DeleteIcon } from "@/components/delete_icon";
import { EditIcon } from "@/components/edit_icon";
import { EyeIcon } from "@/components/eye_icon";
import PlusCircleIcon from "@/components/plus_circle";
import { useUser } from "@auth0/nextjs-auth0/client";
import {
  Modal,
  ModalBody,
  ModalContent,
  ModalFooter,
  ModalHeader,
  Pagination,
  Spacer,
  Table,
  TableBody,
  TableCell,
  TableColumn,
  TableHeader,
  TableRow,
  Tooltip,
  useDisclosure,
} from "@nextui-org/react";
import { Button } from "@nextui-org/react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { projectService } from "./services";

export default function ProjectsPage() {
  const { user, error, isLoading } = useUser();
  const { isOpen, onOpen, onOpenChange } = useDisclosure();
  const [projectsLoading, setProjectsLoading] = useState(true);
  const [projects, setProjects] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalRecords, setTotalRecords] = useState(0);
  const recordsPerPage = parseInt(process.env.NEXT_PUBLIC_DEFAULT_TABLE_ROWS);
  const [recordToDelete, setRecordToDelete] = useState(null);

  const totalPages = useMemo(() => {
    return totalRecords ? Math.ceil(totalRecords / recordsPerPage) : 0;
  }, [totalRecords]);

  useEffect(() => {
    if (recordToDelete == null) {
      let offset = recordsPerPage * (currentPage - 1);
      setProjectsLoading(true);
      projectService.list(recordsPerPage, offset).then((results) => {
        setTotalRecords(results["totalRecords"]);
        setProjects(results["data"]);
        setProjectsLoading(false);
      });
    }
  }, [currentPage, recordToDelete]);

  const onDeleteClick = (e, itemId) => {
    setRecordToDelete(itemId);
    onOpen();
  };

  const handleDelete = (onClose) => {
    projectService.delete(recordToDelete).then((res) => {
      if (!res?.error) {
        onClose();
        setRecordToDelete(null);
      }
    });
  };

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
        <Table
          aria-label=""
          bottomContentPlacement="outside"
          bottomContent={
            projects.length > 0 ? (
              <div className="flex w-full justify-center">
                <Pagination
                  isCompact
                  showControls
                  showShadow
                  color="primary"
                  page={currentPage}
                  total={totalPages}
                  onChange={(page) => setCurrentPage(page)}
                />
              </div>
            ) : null
          }
        >
          <TableHeader>
            <TableColumn className="min-w-48">NAME</TableColumn>
            <TableColumn>TOTAL BINS</TableColumn>
            <TableColumn>INDOOR BINS</TableColumn>
            <TableColumn>OUTDOOR BINS</TableColumn>
            <TableColumn>LAST UPDATED</TableColumn>
            <TableColumn>ACTIONS</TableColumn>
          </TableHeader>
          <TableBody items={projects} emptyContent={projectsLoading ? "Loading..." : "No rows to display."}>
            {(item) => (
              <TableRow key={item.id}>
                <TableCell>
                  <Link className="text-cyan-500 hover:text-indigo-900 font-bold" href={`/projects/${item.id}`}>
                    {item.name}
                  </Link>
                </TableCell>
                <TableCell>{item?.totalBins ?? "-"}</TableCell>
                <TableCell>{item?.indoorBins ?? "-"}</TableCell>
                <TableCell>{item?.outdoorBins ?? "-"}</TableCell>
                <TableCell>{item.last_updated_at_utc}</TableCell>
                <TableCell>
                  <div className="relative flex items-center gap-2">
                    <Tooltip content="View Project">
                      <span className="text-lg text-default-400 cursor-pointer active:opacity-50">
                        <Link href={`/projects/${item.id}`}>
                          <EyeIcon />
                        </Link>
                      </span>
                    </Tooltip>
                    <Tooltip content="Edit Project">
                      <span className="text-lg text-default-400 cursor-pointer active:opacity-50">
                        <Link href={`/projects/${item.id}/edit`}>
                          <EditIcon />
                        </Link>
                      </span>
                    </Tooltip>
                    <Tooltip color="danger" content="Delete Project">
                      <span className="text-lg text-danger cursor-pointer active:opacity-50">
                        <DeleteIcon onClick={(e) => onDeleteClick(e, item.id)} />
                      </span>
                    </Tooltip>
                  </div>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        <Modal isOpen={isOpen} onOpenChange={onOpenChange}>
          <ModalContent>
            {(onClose) => (
              <>
                <ModalHeader className="flex flex-col gap-1">Warning</ModalHeader>
                <ModalBody>
                  <p>
                    The following action will delete the design project and all of its associated bins and routes. Do
                    you still wish to proceed?
                  </p>
                </ModalBody>
                <ModalFooter>
                  <Button color="primary" variant="light" onPress={onClose}>
                    No
                  </Button>
                  <Button color="danger" onPress={(e) => handleDelete(onClose)}>
                    Yes
                  </Button>
                </ModalFooter>
              </>
            )}
          </ModalContent>
        </Modal>
      </div>
    )
  );
}
