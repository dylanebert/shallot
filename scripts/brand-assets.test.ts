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

// Independently enumerated current-app roots; historical install-compat fixtures are not apps.
const expectedDefaults = [
    "bench",
    ..."animate-with-clips annotate-the-world billboards-and-sprites breakable-joints build-a-scene compute-and-readback custom-material day-night-sky drive-a-vehicle first-person game-loop gpu-particles import-a-model joints measure-performance moving-platform overlay-ui physics-playground play-sound ragdoll render-to-a-terminal respond-to-input save-and-restore stylize-the-look surface-friction"
        .split(" ")
        .map((name) => `examples/recipes/${name}`),
    ..."ascii collapse ocean roads sandbox visualization voxel"
        .split(" ")
        .map((name) => `examples/showcase/${name}`),
    ..."blank survive-reload ui-containment"
        .split(" ")
        .map((name) => `packages/shallot/tests/flows/${name}`),
    "packages/shallot/tests/orbit-touch",
]
    .map((path) => `${path}/public/icon.svg`)
    .sort();

test("default icon membership reaches all 37 retained app icons", () => {
    expect(expectedDefaults.length).toBe(37);
    expect(iconTargets().sort()).toEqual(expectedDefaults);
});

test("every default example icon is the rendered mark", () => {
    const targets = iconTargets();
    expect(targets.length).toBeGreaterThan(30);
    for (const file of targets) expect(read(file)).toBe(`${icon()}\n`);
});

test("a project's own icon stays its own", () => {
    const own = "packages/shallot/tests/flows/no-walls/public/icon.svg";
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
