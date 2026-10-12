import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { renderBrandData } from "./branding";

test("generated loading mark data matches its branding assets", () => {
    const generated = resolve(import.meta.dir, "../src/standard/loading/brand-data.ts");
    expect(readFileSync(generated, "utf8")).toBe(renderBrandData());
});
