#!/usr/bin/env bun
import { collectPopulation, readQuarantine, readSurface } from "../src/harness/surface";

const args = Bun.argv.slice(2);
const rootIndex = args.indexOf("--root");
const root = args[rootIndex + 1] ?? process.cwd();
const violations = readSurface(root);
for (const violation of violations) console.error(violation);
if (violations.length > 0) process.exit(1);
const population = collectPopulation(root);
console.log(
    `${population.rows.length} declared checks (${readQuarantine(root).rows.length} quarantined)`,
);
