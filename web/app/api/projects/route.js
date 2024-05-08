"server only";

/**
 * API route for project search requests.
 *
 * References:
 * - https://nextjs.org/docs/app/api-reference/functions/next-request
 * - https://nextjs.org/docs/app/api-reference/functions/next-response
 */
import prisma from "@/services/db";
import { NextRequest, NextResponse } from "next/server";

/**
 * Creates a new project.
 *
 *  @param {NextRequest} - The HTTP request.
 */
export async function POST(request) {
  let timestamp = new Date().toISOString();
  let requestBody = await request.json();
  let project = await prisma.foodware_project.create({
    data: {
      name: requestBody["name"],
      description: requestBody["description"],
      locale_id: requestBody["geography"],
      created_at_utc: timestamp,
      last_updated_at_utc: timestamp,
    },
  });
  return NextResponse.json(project);
}

/**
 * Lists projects using pagination.
 *
 *  @param {NextRequest} - The HTTP request. Contains a geography id.
 */
export async function GET(request, { params }) {
  let searchParams = request.nextUrl.searchParams;
  let results = await prisma.foodware_project.findMany({
    skip: parseInt(searchParams.get("offset")),
    take: parseInt(searchParams.get("limit")),
    orderBy: [{ name: "asc" }],
  });
  let totalRecords = await prisma.foodware_project.count();
  return NextResponse.json({
    data: results,
    totalRecords: totalRecords,
  });
}
