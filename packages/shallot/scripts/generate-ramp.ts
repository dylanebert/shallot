// Generates `extras/cells/ramp-table.ts`: the coverage-ordered fill glyph ramp, sorted
// by measured ink coverage, not a hand-authored character set in code-point order. Renders each
// candidate printable-ASCII glyph's outline against the cells face (`assets/jetbrains-mono.ttf`, the house
// monospace) and measures its ink coverage: the glyph's TrueType contour area (Green's theorem over
// the parsed outline, `extras/text/font.ts`'s own `glyphPath`) divided by the font's em-square area,
// sorted ascending.
//
// Lives beside `gen-tumble-fixtures.ts` / `gen-tumble-gold.ts` (`packages/shallot/scripts/`), not under
// `src/extras/cells/` — a generator that reaches across module boundaries (`extras/text/font.ts`) and
// touches Node-only `readFileSync`/`writeFileSync` is exactly the shape those two carve out for
// themselves, and the same reasoning applies here: `check-imports.ts` only walks `src/`, so a script
// tree is where cross-module tooling lives rather than a deep import `ramp.ts`'s own barrel would have to
// carry into the browser-importable graph. `computeRampTable` and `glyphCoverage` are exported pure
// functions so `extras/cells/ramp-table.test.ts` can call the identical derivation the committed table
// came from and assert they still agree — the arm that makes `ramp-table.ts` a reproducible derivation
// rather than a hand-authored guess with a script beside it.
//
// The directional glyphs (`CELL_DIRECTIONAL_GLYPHS`, `ramp.ts`) are excluded from the fill candidates —
// they're selected by edge angle, not ink coverage (`ramp.ts`'s own module doc), so mixing one into the
// coverage-sorted fill ramp would double-book it under a rule that doesn't govern it. The curated curved
// marks (`CELL_FILL_EXCLUDED_GLYPHS`, `ramp.ts`) are excluded too — the s3r fill-treatment amendment's
// "angular glyphs only, never curved ones" — hand-curated rather than a coverage or curvature threshold
// (`ramp.ts`'s own docblock has the derivation for why a threshold would wrongly exclude glyphs the
// reference itself keeps).
//
// Run from the shallot repo root: `bun run packages/shallot/scripts/generate-ramp.ts`. Nothing
// regenerates `ramp-table.ts` automatically — review the diff before committing, the same discipline
// `gen-tumble-fixtures.ts` documents for its own committed output.

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { CELL_DIRECTIONAL_GLYPHS, CELL_FILL_EXCLUDED_GLYPHS } from "../src/extras/cells/ramp";
import { type Font, parseFont } from "../src/extras/text/font";

const FONT_URL = new URL("../../../assets/jetbrains-mono.ttf", import.meta.url);
const OUTPUT_URL = new URL("../src/extras/cells/ramp-table.ts", import.meta.url);

/** one point in a flattened glyph contour, font units. */
interface Point {
    x: number;
    y: number;
}

function candidateChars(): string[] {
    const excluded = new Set<string>([...CELL_DIRECTIONAL_GLYPHS, ...CELL_FILL_EXCLUDED_GLYPHS]);
    const chars: string[] = [];
    for (let code = 0x20; code <= 0x7e; code++) {
        const ch = String.fromCharCode(code);
        if (!excluded.has(ch)) chars.push(ch);
    }
    return chars;
}

// Green's theorem: the signed area of one closed, already-flattened contour — the shoelace sum over
// consecutive vertex pairs.
function contourArea(points: readonly Point[]): number {
    let sum = 0;
    for (let i = 0; i < points.length; i++) {
        const a = points[i];
        const b = points[(i + 1) % points.length];
        sum += a.x * b.y - b.x * a.y;
    }
    return sum / 2;
}

// Chord-flatten one quadratic Bezier segment (`font.ts`'s own `Q cx,cy,x,y` emission) into STEPS points,
// t in (0, 1] — t=0 (the segment's start) is already the caller's running cursor. STEPS=8's flattening
// error is NOT well inside every adjacent pair's coverage gap in the committed table — measured directly
// against the committed table (this repair round): the smallest adjacent gap is exactly 0 (an exact tie,
// "<" / ">"), the next two smallest are "*" / "+" at ~4e-6 and "H" / "E" at ~2.1e-5, and 6 of the table's
// 90 adjacent gaps sit below 3e-5 — smaller than the ~5.4e-5 largest single-glyph coverage delta measured
// between STEPS=8 and STEPS=32 (re-measured this round). So the margin claim was false as stated.
//
// What holds instead, also re-measured this round: STEPS=8 and STEPS=32 still produce an identical sort
// order end to end, ties included. The defensible reading is that a pair whose coverage differs by
// low-1e-5 or less is effectively tied at this measurement's resolution, and this ramp's contract
// (`ramp.ts`'s module doc, the Locked decision) only needs coverage-ordered *bands* for a tone-based
// fallback selector, not a strict total order at deltas this small — two glyphs this close in ink density
// read as interchangeable to the selector they feed. If a future STEPS or font change ever does flip one
// of these near-ties, nothing downstream regresses on that basis alone; it would regress only if the flip
// crossed a coverage gap large enough to matter to the selector, which is a materially different claim
// than "produces an identical sort order" and isn't the one this comment is making.
const STEPS = 8;
function flattenQuadratic(p0: Point, c: Point, p1: Point): Point[] {
    const pts: Point[] = [];
    for (let i = 1; i <= STEPS; i++) {
        const t = i / STEPS;
        const mt = 1 - t;
        pts.push({
            x: mt * mt * p0.x + 2 * mt * t * c.x + t * t * p1.x,
            y: mt * mt * p0.y + 2 * mt * t * c.y + t * t * p1.y,
        });
    }
    return pts;
}

// Parse `font.ts`'s own path serialization (one or more `M x,y (L x,y | Q cx,cy,x,y)* Z` contours back to
// back) into flattened point-list contours, ready for `contourArea`.
function parseContours(path: string): Point[][] {
    const contours: Point[][] = [];
    const tokens = path.match(/[MLQZ][^MLQZ]*/g) ?? [];
    let current: Point[] = [];
    let cursor: Point = { x: 0, y: 0 };
    for (const token of tokens) {
        const cmd = token[0];
        const nums = token
            .slice(1)
            .split(",")
            .filter((s) => s.length > 0)
            .map(Number);
        if (cmd === "M") {
            cursor = { x: nums[0], y: nums[1] };
            current = [cursor];
        } else if (cmd === "L") {
            cursor = { x: nums[0], y: nums[1] };
            current.push(cursor);
        } else if (cmd === "Q") {
            const c = { x: nums[0], y: nums[1] };
            const p1 = { x: nums[2], y: nums[3] };
            current.push(...flattenQuadratic(cursor, c, p1));
            cursor = p1;
        } else if (cmd === "Z") {
            contours.push(current);
        }
    }
    return contours;
}

/**
 * ink coverage for one glyph: `|sum of every contour's signed area| / em-square area`. Summing every
 * contour's *signed* area before taking the absolute value (rather than summing each contour's own
 * `|area|`) is what makes a counter — the bowl of `e`, `o`, the hole of `@` — net-subtract instead of
 * over-counting: TrueType winds a hole opposite to its outer contour (the nonzero fill rule), so the
 * signed sum already nets them the way a rasterizer's ink coverage would. A blank glyph (space —
 * `font.glyphPath` returns `null` for an empty contour list) has zero coverage; a code point outside the
 * font's cmap resolves through `font.ts`'s own `.notdef` fallback (glyph id 0), whatever that glyph's
 * outline measures — never a thrown error.
 */
export function glyphCoverage(font: Font, char: string): number {
    const path = font.glyphPath(char);
    if (!path) return 0;
    const contours = parseContours(path);
    let total = 0;
    for (const c of contours) total += contourArea(c);
    return Math.abs(total) / (font.unitsPerEm * font.unitsPerEm);
}

/** one fill-ramp row: a candidate glyph plus its measured ink coverage. */
export interface RampEntry {
    readonly char: string;
    readonly coverage: number;
}

/**
 * the coverage-ordered fill table: every printable-ASCII candidate minus the curated directional set
 * ({@link CELL_DIRECTIONAL_GLYPHS}), sorted ascending by {@link glyphCoverage} — ties (a shared coverage
 * reading) broken by code point for a deterministic, re-derivable order. Pure over `font`, so the
 * generator's `main()` and `ramp-table.test.ts`'s reproduction check both call this against the same font
 * bytes and must agree.
 */
export function computeRampTable(font: Font): RampEntry[] {
    return candidateChars()
        .map((char) => ({ char, coverage: glyphCoverage(font, char) }))
        .sort((a, b) => a.coverage - b.coverage || a.char.codePointAt(0)! - b.char.codePointAt(0)!);
}

/** load the cells face (`assets/jetbrains-mono.ttf`) the way `extras/text/font.test.ts` does — Node-only
 *  (`readFileSync`), so callers stay confined to this script and its own test, never `ramp.ts`. */
export function loadBrandFont(): Font {
    const bytes = readFileSync(fileURLToPath(FONT_URL));
    const buffer = bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
    ) as ArrayBuffer;
    return parseFont(buffer);
}

function renderTable(table: readonly RampEntry[]): string {
    // Full precision (not `.toFixed`): `ramp-table.test.ts` compares this committed table against a
    // fresh `computeRampTable` call byte-for-byte, so a truncated literal here would fail its own
    // reproduction proof against the untruncated value the generator actually computed.
    const rows = table
        .map(
            (r) =>
                `    { char: ${JSON.stringify(r.char)}, coverage: ${JSON.stringify(r.coverage)} },`,
        )
        .join("\n");
    return `// GENERATED by scripts/generate-ramp.ts — do not hand-edit. Re-run \`bun run
// packages/shallot/scripts/generate-ramp.ts\` from the shallot repo root and review the diff.
// The coverage-ordered fill glyph ramp (module doc, scripts/generate-ramp.ts): every printable-ASCII
// candidate minus the directional set (ramp.ts's CELL_DIRECTIONAL_GLYPHS), sorted ascending by measured
// ink coverage against the cells face (assets/jetbrains-mono.ttf). \`ramp-table.test.ts\` proves this file still
// equals a fresh computation off the same font.

export interface RampEntry {
    readonly char: string;
    readonly coverage: number;
}

export const RAMP_TABLE: readonly RampEntry[] = [
${rows}
];
`;
}

function main(): void {
    const table = computeRampTable(loadBrandFont());
    writeFileSync(fileURLToPath(OUTPUT_URL), renderTable(table));
    console.log(`wrote ${table.length} entries to ${fileURLToPath(OUTPUT_URL)}`);
}

if (import.meta.main) main();
