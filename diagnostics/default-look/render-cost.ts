import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { costReport } from "./cost-report";

const input = process.argv[2];
if (!input) throw new Error("Usage: bun diagnostics/default-look/render-cost.ts <cost.json>");
const rows = JSON.parse(await readFile(input, "utf8"));
const output = resolve(import.meta.dir, "../../test-results/default-look/cost.html");
await mkdir(resolve(output, ".."), { recursive: true });
await writeFile(output, costReport(rows));
console.log(`Cost page: ${output}`);
