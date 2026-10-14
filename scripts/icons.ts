import { mkdirSync, readdirSync, writeFileSync } from "node:fs";
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
    const targets = readdirSync(resolve(ROOT, "examples")).map(
        (name) => `examples/${name}/public/icon.svg`,
    );
    if (targets.length === 0) throw new Error("no example icons; the write is empty");
    for (const file of targets) {
        mkdirSync(resolve(ROOT, file, ".."), { recursive: true });
        writeFileSync(resolve(ROOT, file), svg);
    }
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
    const png = toPng(framed, branding.palettes.dark, 16, branding.palettes.dark.bg);
    // ICO permits PNG entries; zero dimensions in its directory denote 256 pixels.
    const header = Buffer.alloc(22);
    header.writeUInt16LE(1, 2);
    header.writeUInt16LE(1, 4);
    header.writeUInt16LE(1, 10);
    header.writeUInt16LE(32, 12);
    header.writeUInt32LE(png.length, 14);
    header.writeUInt32LE(22, 18);
    writeFileSync(resolve(ROOT, "crates/native/icon.ico"), Buffer.concat([header, png]));
    return targets.length;
}

if (import.meta.main) console.log(`icons: wrote ${icons()} example icons and ${NATIVE_ICON}`);
