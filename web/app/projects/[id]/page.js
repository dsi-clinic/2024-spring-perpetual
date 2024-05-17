"use client";

import { CheckIcon } from "@/components/check_icon";
import { BinMap, LocaleMap } from "@/components/map";
import { PencilIcon } from "@/components/pencil_icon";
import { SearchIcon } from "@/components/search_icon";
import Stat from "@/components/stat";
import { get } from "@/services/http";
import {
  Button,
  Divider,
  Dropdown,
  DropdownItem,
  DropdownMenu,
  DropdownTrigger,
  Input,
  Pagination,
  Spacer,
  Spinner,
  Table,
  TableBody,
  TableCell,
  TableColumn,
  TableHeader,
  TableRow,
} from "@nextui-org/react";
import "maplibre-gl/dist/maplibre-gl.css";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

export default function ProjectPage({ params }) {
  const [project, setProject] = useState(null);
  const [projectLoading, setProjectLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(0);
  const recordsPerPage = 10;
  const [filterValue, setFilterValue] = useState("");
  const [sortDescriptor, setSortDescriptor] = useState({
    column: "name",
    direction: "ascending",
  });

  const getProject = async (id) => {
    const url = `/api/projects/${id}`;
    const errMsg = `Failed to fetch project ${id}.`;
    return await get(url, errMsg);
  };

  useEffect(() => {
    getProject(params.id).then((data) => {
      setProject(data);
      setProjectLoading(false);
      setPages(Math.ceil(data.bins.length / recordsPerPage));
    });
  }, []);

  const onSearchChange = useCallback((value) => {
    if (value) {
      setFilterValue(value);
      setPage(1);
    } else {
      setFilterValue("");
    }
  }, []);

  const onClear = useCallback(() => {
    setFilterValue("");
    setPage(1);
  }, []);

  const items = useMemo(() => {
    if (project === null) return [];

    let eligibleBins =
      filterValue === ""
        ? project.bins
        : project.bins.filter((b) => {
            return b.name.toLowerCase().includes(filterValue.toLowerCase());
          });

    let sortedEligibleBins = eligibleBins.sort((a, b) => {
      const first = a[sortDescriptor.column];
      const second = b[sortDescriptor.column];
      const cmp = first < second ? -1 : first > second ? 1 : 0;
      return sortDescriptor.direction === "descending" ? -cmp : cmp;
    });

    setPages(sortedEligibleBins.length);

    const start = (page - 1) * recordsPerPage;
    const end = start + recordsPerPage;
    return sortedEligibleBins.slice(start, end) ?? [];
  }, [page, project, filterValue, sortDescriptor]);

  if (projectLoading) {
    return <Spinner label="Loading..." color="primary" labelColor="primary" size="large" />;
  }

  return (
    <div className="flex w-full flex-col flex-grow px-10">
      <h2 className="from-black to-stone-500 text-3xl font-bold tracking-[-0.02em]">
        {project.name}{" "}
        <Link href={`/projects/${params.id}/edit`}>
          <PencilIcon size={5} fillColor="rgb(6 182 212)" />
        </Link>
      </h2>
      {project.description && (
        <>
          <p className="text-foreground-500 text-base">{project.description}</p>
          <Spacer y={4} />
        </>
      )}
      <Spacer y={3} />
      <h3 className="text-slate-600 text-lg font-bold tracking-[-0.02em]">Instructions</h3>
      <Divider className="mb-5" />
      <div className="space-y-2">
        <p>
          Local points of interest such as restaurants, big box grocery stores, and large hotels and apartments were
          programmatically fetched from Google Maps and then, using a rule-based algorithm, excluded from the project or
          designated as an indoor or outdoor bin location. Confirm the automated bin selections by marking bins as
          excluded or included using the data table or map. You can also sort table records by column and filter records
          using the search bar.
        </p>
        <p>
          Once you are done with your edits, click on the "Workshop" button to generate a new map on the Padlet
          platform. There, Perpetual staff, partner organizations, and community members can like and comment on
          locations, delete locations, and propose new locations. During this workshop period, the data table becomes
          read-only and syncs with the Padlet board upon page refresh or on demand by clicking the "Update from Padlet"
          button. After the public workshops have finished, click on "Complete" button to finalize the set of bins and
          move onto the route-planning stage.
        </p>
      </div>
      <Spacer y={5} />
      <h3 className="text-slate-600 text-lg font-bold tracking-[-0.02em]">Bin Locations</h3>
      <Divider className="mb-5" />
      <div className="flex flex-row gap-x-10">
        <Stat value={project.bins.length.toLocaleString()} units="Total # Bins" />
        <Stat value={project.bins.filter((b) => b.classification === "Indoor").length} units="Indoor Bins" />
        <Stat value={project.bins.filter((b) => b.classification === "Outdoor").length} units="Outdoor Bins" />
      </div>
      <Spacer y={5} />
      <div className="grid grid-cols-2 gap-4">
        <Table
          aria-label="Bins table"
          selectionMode="multiple"
          topContent={
            <div>
              <Input
                isClearable
                className="w-full sm:max-w-[66%]"
                placeholder="Search by name..."
                startContent={<SearchIcon />}
                value={filterValue}
                onClear={() => onClear()}
                onValueChange={onSearchChange}
              />
              <Spacer y={2} />
              <Dropdown>
                <DropdownTrigger>
                  <Button variant="bordered">Mark As</Button>
                </DropdownTrigger>
                <DropdownMenu aria-label="Static Actions">
                  <DropdownItem key="new">Included</DropdownItem>
                  <DropdownItem key="copy">Excluded</DropdownItem>
                </DropdownMenu>
              </Dropdown>
            </div>
          }
          bottomContent={
            <div className="flex w-full justify-center">
              <Pagination
                isCompact
                showControls
                showShadow
                color="secondary"
                page={page}
                total={pages}
                onChange={(page) => setPage(page)}
              />
            </div>
          }
          sortDescriptor={sortDescriptor}
          onSortChange={setSortDescriptor}
        >
          <TableHeader>
            <TableColumn key="status">STATUS</TableColumn>
            <TableColumn key="bin type" allowsSorting>
              BIN TYPE
            </TableColumn>
            <TableColumn key="category" allowsSorting>
              CATEGORY
            </TableColumn>
            <TableColumn key="name" allowsSorting>
              NAME
            </TableColumn>
            <TableColumn>ADDRESS</TableColumn>
          </TableHeader>
          <TableBody items={items}>
            {(item) => (
              <TableRow key={item.id}>
                <TableCell>
                  <CheckIcon fill="green" strokeColor="white" />
                </TableCell>
                <TableCell>{item.classification}</TableCell>
                <TableCell>{item.parent_category_name}</TableCell>
                <TableCell>{item.name}</TableCell>
                <TableCell>{item.formatted_address}</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        {project.bins ? (
          <BinMap boundary={project.boundary} bins={project.bins} />
        ) : (
          <LocaleMap boundary={project.boundary} />
        )}
      </div>
      <Spacer y={3} />
    </div>
  );
}
