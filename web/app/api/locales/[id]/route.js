"server only";

/**
 * API route to fetch a locale by id.
 *
 * References:
 * - https://nextjs.org/docs/app/api-reference/functions/next-request
 * - https://nextjs.org/docs/app/api-reference/functions/next-response
 */
import prisma from "@/services/db";
import { NextRequest, NextResponse } from "next/server";

/**
 * Fetches a given geography based on id.
 *
 *  @param {NextRequest} - The HTTP request. Contains a geography id.
 */
export async function GET(request, { params }) {
  let { id } = params;
  let geometry = await prisma.$queryRaw`
      SELECT id, name, ST_ASGeoJSON(geometry) AS data
      FROM locale
      WHERE id = ${parseInt(id)};
  `;
  if (!!geometry.length) {
    return NextResponse.json({
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: JSON.parse(geometry[0]["data"]),
          properties: {
            id: geometry[0]["id"],
            name: geometry[0]["name"],
          },
        },
      ],
    });
  }
  return new Response("Locale not found.", {
    status: 404,
  });
}
