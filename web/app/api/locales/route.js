"server only";

/**
 * API route for geography search requests.
 *
 * References:
 * - https://nextjs.org/docs/app/api-reference/functions/next-request
 * - https://nextjs.org/docs/app/api-reference/functions/next-response
 */
import prisma from "@/services/db";
import { NextRequest, NextResponse } from "next/server";

/**
 * Searches for a geography by name and returns the
 * top results as a list of objects with "id" and "name"
 * properties. Full-text search is achieved in PostgreSQL
 * using a combination of similarity scores calculated
 * from trigrams ("pg_trgm") and ranks calculated from
 * normalized vectors of lexemes ("tsvector").
 *
 *  @param {NextRequest} - The HTTP request. Contains the
 *      search term and the number of matches to return.
 */
export async function POST(request) {
  let { searchPhrase, limit } = await request.json();
  let formattedSearchPhrase = searchPhrase.replace(/\s+/g, " ").trim();
  let tsquerySearchPhrase = formattedSearchPhrase.replace(/\s+/g, " & ");

  let searchResults = await prisma.$queryRaw`
    SELECT
      id,
      name
    FROM
      (
        SELECT 
          id::varchar(255), 
          name,
          ts_rank(
            name_vector, 
            to_tsquery('english', ${tsquerySearchPhrase})
          ) AS rank,
          similarity(${formattedSearchPhrase}, name) AS sml
        FROM locale
        ORDER BY sml DESC, rank DESC, name ASC
        LIMIT ${limit}
      ) AS results
      WHERE rank > 1e-20 OR sml > 0;
    `;
  return NextResponse.json(searchResults);
}
