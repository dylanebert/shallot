import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { compose, DARK, fromBlocks, MARK, toSvg } from "../src/standard/loading/mark";
import { toPng } from "./png";

// Every default icon a shallot project ships is a render of the one bitmap mark, so the shape
// can't drift between the boot splash, the brand page and a scaffolded project's favicon.
// `--write` regenerates them; `brand-assets.test.ts` asserts byte equality, which makes drift a
// red rather than something a reader has to notice.

export const ROOT = resolve(import.meta.dir, "..");

/** The favicon: the canonical mark at one pixel per cell, dark-theme hexes, transparent ground. */
export function icon(): string {
    return toSvg(fromBlocks(MARK.m), DARK, 1);
}

// The native window icon is opaque and square: the 12×14 mark centred in a 16×16 field on the
// dark ground, 64 device pixels a cell. A window manager scales it down, so the frame is what
// keeps the mark off the edge at small sizes.
const FRAME = 16;
const NATIVE_SCALE = 64;

/** The native window icon: the framed mark on the dark ground, 1024×1024 PNG bytes. */
export function nativeIcon(): Uint8Array {
    const mark = fromBlocks(MARK.m);
    const width = mark[0]?.length ?? 0;
    const framed = compose(FRAME, FRAME, [
        {
            grid: mark,
            x: Math.floor((FRAME - width) / 2),
            y: Math.floor((FRAME - mark.length) / 2),
        },
    ]);
    return toPng(framed, DARK, NATIVE_SCALE, DARK.bg);
}

/** Tracked example icons that carry the default, relative to the repo root. */
export function iconTargets(): string[] {
    const tracked = Bun.spawnSync(["git", "ls-files", "-z", "examples"], { cwd: ROOT });
    if (!tracked.success) throw new Error("brand-assets: `git ls-files` failed");
    const files = tracked.stdout
        .toString()
        .split("\0")
        .filter((file) => file.endsWith("/public/icon.svg"));
    if (files.length === 0) throw new Error("brand-assets: no example icons — the check is empty");
    return files;
}

export const NATIVE_ICON = "assets/icon-1024.png";

if (import.meta.main) {
    if (!process.argv.includes("--write")) {
        console.error("usage: bun run scripts/brand-assets.ts --write");
        process.exit(1);
    }
    const svg = `${icon()}\n`;
    const targets = iconTargets();
    for (const file of targets) writeFileSync(resolve(ROOT, file), svg);
    writeFileSync(resolve(ROOT, NATIVE_ICON), nativeIcon());
    console.log(`✓ brand assets written (${targets.length} icons, native icon)`);
}
