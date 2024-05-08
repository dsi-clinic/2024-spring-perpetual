/**
 * Provides database access using the PrismaClient ORM.
 * Uses a singleton in non-production environments to prevent
 * new clients from being instantiated during hot-reloading.
 *
 * References:
 * - https://stackoverflow.com/a/72537459
 * - https://www.prisma.io/docs/guides/other/troubleshooting-orm/help-articles/nextjs-prisma-client-dev-practices
 */
import { PrismaClient } from "@prisma/client";

let prisma;

if (process.env.NODE_ENV === "production") {
  prisma = new PrismaClient();
} else {
  globalThis.prisma = globalThis.prisma || new PrismaClient();
  prisma = globalThis.prisma;
}

export default prisma;
