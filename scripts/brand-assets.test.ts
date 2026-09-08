// The shipped icons are generated, so their arms are byte equality against the renderer: a hand
// edit, a half-applied regeneration or a mark change that skipped `--write` all read red here.

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { template } from "../packages/create-shallot/index";
import {
    icon,
    iconTargets,
    NATIVE_ICON,
    nativeIcon,
    ROOT,
    SCAFFOLD,
    scaffoldSource,
} from "./brand-assets";

const read = (file: string) => readFileSync(resolve(ROOT, file), "utf8");

test("every default example icon is the rendered mark", () => {
    const targets = iconTargets();
    expect(targets.length).toBeGreaterThan(30);
    for (const file of targets) expect(read(file)).toBe(`${icon()}\n`);
});

test("a project's own icon stays its own", () => {
    const own = "examples/flows/no-walls/public/icon.svg";
    expect(iconTargets()).not.toContain(own);
    expect(read(own)).toContain('fill="#f233b3"');
});

test("the scaffold writes the same icon", () => {
    expect(template("demo")["public/icon.svg"]).toBe(`${icon()}\n`);
    expect(scaffoldSource(read(SCAFFOLD))).toBe(read(SCAFFOLD));
});

test("the native window icon is the framed mark at 1024", () => {
    const bytes = nativeIcon();
    const header = new DataView(bytes.buffer, bytes.byteOffset);
    expect(header.getUint32(16)).toBe(1024);
    expect(header.getUint32(20)).toBe(1024);
    expect(readFileSync(resolve(ROOT, NATIVE_ICON))).toEqual(Buffer.from(bytes));
});
