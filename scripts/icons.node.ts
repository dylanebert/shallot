import { expect, setDefaultTimeout, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { CEILING } from "@dylanebert/shallot/testing";
import { compose, fromBlocks, toSvg } from "../src/standard/loading/brand";
import { readBranding } from "./branding";
import { toPng } from "./png";

setDefaultTimeout(CEILING.node);
const root = resolve(import.meta.dir, "..");
test("every example owns an editable base-correct standard favicon", () => {
    const branding = readBranding();
    const standard = Buffer.from(
        `${toSvg(fromBlocks(branding.marks.m), branding.palettes.dark, 1)}\n`,
    );
    for (const name of readdirSync(resolve(root, "examples"))) {
        const html = readFileSync(resolve(root, "examples", name, "index.html"), "utf8");
        expect(html).toContain('rel="icon" type="image/svg+xml" href="/icon.svg"');
        expect(readFileSync(resolve(root, "examples", name, "public/icon.svg"))).toEqual(standard);
    }
});

test("native PNG and Windows ICO are generated from the standard framed mark", () => {
    const branding = readBranding();
    const mark = fromBlocks(branding.marks.m);
    const framed = compose(16, 16, [{ grid: mark, x: 2, y: 1 }]);
    const png = (scale: number) =>
        toPng(framed, branding.palettes.dark, scale, branding.palettes.dark.bg);
    expect(readFileSync(resolve(root, "assets/icon-1024.png"))).toEqual(Buffer.from(png(64)));
    const ico = readFileSync(resolve(root, "crates/native/icon.ico"));
    expect([...ico.subarray(0, 14)]).toEqual([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 32, 0]);
    expect(ico.readUInt32LE(18)).toBe(22);
    expect(ico.readUInt32LE(14)).toBe(ico.length - 22);
    expect(ico.subarray(22)).toEqual(Buffer.from(png(16)));
});
