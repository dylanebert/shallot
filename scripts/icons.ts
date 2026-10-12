import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { compose, fromBlocks, toSvg } from "../src/standard/loading/brand";
import { readBranding } from "./branding";
import { toPng } from "./png";

const ROOT = resolve(import.meta.dir, "..");

// Every default icon a shallot project ships is a render of the one bitmap mark, so the shape
// can't drift between the boot splash, a scaffolded project's favicon and the native window.
const NATIVE_ICON = "assets/icon-1024.png";
// The native icon is opaque and square: the 12×14 mark centred in a 16×16 field on the dark ground,
// 64 device pixels a cell. A window manager scales it down; the frame keeps the mark off the edge.
const FRAME = 16;
const NATIVE_SCALE = 64;

function icons(): number {
    const branding = readBranding();
    const mark = fromBlocks(branding.marks.m);
    const svg = `${toSvg(mark, branding.palettes.dark, 1)}\n`;
    const tracked = Bun.spawnSync(["git", "ls-files", "-z", "examples"], { cwd: ROOT });
    if (!tracked.success) throw new Error("`git ls-files` failed");
    const targets = tracked.stdout
        .toString()
        .split("\0")
        .filter((file) => file.endsWith("/public/icon.svg"));
    if (targets.length === 0) throw new Error("no example icons; the write is empty");
    for (const file of targets) writeFileSync(resolve(ROOT, file), svg);
    const width = mark[0]?.length ?? 0;
    const framed = compose(FRAME, FRAME, [
        {
            grid: mark,
            x: Math.floor((FRAME - width) / 2),
            y: Math.floor((FRAME - mark.length) / 2),
        },
    ]);
    writeFileSync(
        resolve(ROOT, NATIVE_ICON),
        toPng(framed, branding.palettes.dark, NATIVE_SCALE, branding.palettes.dark.bg),
    );
    return targets.length;
}

if (import.meta.main) console.log(`icons: wrote ${icons()} example icons and ${NATIVE_ICON}`);
