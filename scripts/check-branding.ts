import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { renderBrandData } from "./branding";

const generated = resolve(import.meta.dir, "../src/standard/loading/brand-data.ts");
if (readFileSync(generated, "utf8") !== renderBrandData()) {
    console.error("✗ generated loading mark data is stale; run `bun scripts/branding.ts`");
    process.exit(1);
}
console.log("✓ loading mark data matches assets/branding");
