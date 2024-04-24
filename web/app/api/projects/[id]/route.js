"server only";

/**
 * API route for projet search requests.
 *
 * References:
 * - https://nextjs.org/docs/app/api-reference/functions/next-request
 * - https://nextjs.org/docs/app/api-reference/functions/next-response
 */
import prisma from "@/services/db";
import { NextRequest, NextResponse } from "next/server";

/**
 * Fetches a project by id.
 *
 *  @param {NextRequest} - The HTTP request. Contains a geography id.
 */
export async function GET(request, { params }) {
  let { id } = params;
  let results = await prisma.$queryRaw`
    SELECT
      proj.id,
      proj.name,
      proj.description,
      proj.created_at_utc,
      proj.last_updated_at_utc,
      locale.id AS locale_id,
      locale.name AS locale_name,
      ST_ASGeoJSON(locale.geometry) AS boundary
    FROM foodware_project as proj
    JOIN locale ON proj.locale_id = locale.id
    WHERE proj.id = ${parseInt(id)};
  `;

  if (!results.length) {
    return new Response("Project not found.", {
      status: 404,
    });
  }

  let project = results[0];
  let parsedProject = {
    id: project["id"],
    name: project["name"],
    description: project["description"],
    locale_id: project["locale_id"],
    locale_name: project["locale_name"],
    created_at_utc: project["created_at_utc"],
    last_updated_at_utc: project["last_updated_at_utc"],
    boundary: {
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: JSON.parse(project["boundary"]) }],
    },
  };
  return NextResponse.json(parsedProject);
}

/**
 * Updates a project by id.
 *
 *  @param {NextRequest} - The HTTP request. Contains a geography id.
 */
export async function PATCH(request, { params }) {
  let id = parseInt(params.id);
  let { name, description, geography } = await request.json();
  let timestamp = new Date().toISOString();
  const updatedProject = await prisma.foodware_project.update({
    where: {
      id: id,
    },
    data: {
      name: name,
      description: description,
      locale_id: geography,
      last_updated_at_utc: timestamp,
    },
  });
  return NextResponse.json(updatedProject);
}

/**
 * Deletes a project and its associations.
 *
 *  @param {NextRequest} - The HTTP request. Contains a geography id.
 */
export async function DELETE(request, { params }) {
  let id = parseInt(params.id);
  const deletedProject = await prisma.foodware_project.delete({
    where: {
      id: id,
    },
  });
  return NextResponse.json(deletedProject);
}
