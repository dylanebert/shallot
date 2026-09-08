// The shallot mark, wordmark and splash as data plus pure renderers. Browser-safe: no node
// imports, so the same module draws the brand page, prints the CLI banner and emits the
// downloadable SVGs.
//
// The source of truth is a square-pixel bitmap. A terminal prints it as half blocks (one
// character is two stacked pixels, square at the 1:2 cell every terminal ships); an image draws
// it as squares. Both come from the one bitmap, so the shape never drifts between surfaces.

/** A pixel color role. Palettes map roles to hex per theme. */
export type Tone = "gold" | "dim" | "ink";

/** Rows of pixels; `null` is transparent. */
export type Grid = (Tone | null)[][];

/** The mark at three sizes, as the half-block strings a terminal prints. `m` is canonical. */
export const MARK = {
    s: ["    ▄▄", "   ▄██▄", "  ▄████▄", "▄████████▄", "▀████████▀", "  ▀▀▀▀▀▀"],
    m: [
        "     ▄▄",
        "    ▄██▄",
        "   ▄████▄",
        " ▄████████▄",
        "████████████",
        "▀██████████▀",
        "  ▀▀▀▀▀▀▀▀",
    ],
    l: [
        "      ▄▄",
        "     ▄██▄",
        "    ▄████▄",
        "  ▄████████▄",
        " ████████████",
        "██████████████",
        " ▀██████████▀",
        "   ▀▀▀▀▀▀▀▀",
    ],
} as const;

/** 5×7 bitmap face, lowercase, the letters the name needs. `#` is ink. */
export const FONT: Record<string, readonly string[]> = {
    s: [".....", ".....", ".####", "#....", ".###.", "....#", "####."],
    h: ["#....", "#....", "####.", "#...#", "#...#", "#...#", "#...#"],
    a: [".....", ".....", ".###.", "....#", ".####", "#...#", ".####"],
    l: ["#..", "#..", "#..", "#..", "#..", "#..", ".##"],
    o: [".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###."],
    t: [".#..", ".#..", "####", ".#..", ".#..", ".#..", "..##"],
};

export const NAME = "shallot";

/** Hex per tone and ground, one set per theme. */
export type Palette = { gold: string; dim: string; ink: string; bg: string };

export const DARK: Palette = { gold: "#d49560", dim: "#7a5a3a", ink: "#f0e6d6", bg: "#141210" };
export const LIGHT: Palette = { gold: "#d49560", dim: "#e6c6a4", ink: "#2a231e", bg: "#f7f3ec" };

/** Half-block rows → bitmap of the given tone. Width is the longest row. */
export function fromBlocks(rows: readonly string[], tone: Tone = "gold"): Grid {
    const width = Math.max(...rows.map((r) => r.length));
    const grid: Grid = [];
    for (const row of rows) {
        const top: (Tone | null)[] = [];
        const bottom: (Tone | null)[] = [];
        for (let x = 0; x < width; x++) {
            const ch = row[x] ?? " ";
            top.push(ch === "█" || ch === "▀" ? tone : null);
            bottom.push(ch === "█" || ch === "▄" ? tone : null);
        }
        grid.push(top, bottom);
    }
    return grid;
}

/** The name set in the bitmap face, one pixel between letters. */
export function word(text: string = NAME, tone: Tone = "ink"): Grid {
    const rows: Grid = Array.from({ length: 7 }, () => []);
    [...text].forEach((ch, i) => {
        const glyph = FONT[ch];
        if (!glyph) throw new Error(`no glyph for "${ch}"`);
        glyph.forEach((line, y) => {
            const row = rows[y] as (Tone | null)[];
            if (i > 0) row.push(null);
            for (const c of line) row.push(c === "#" ? tone : null);
        });
    });
    return rows;
}

/** A placed bitmap. `x` in pixels, `y` in pixels. */
export type Layer = { grid: Grid; x: number; y: number };

/** Stamp layers onto a blank grid, later layers over earlier. Pixel rows must be even for half blocks. */
export function compose(width: number, height: number, layers: readonly Layer[]): Grid {
    const out: Grid = Array.from({ length: height }, () => Array<Tone | null>(width).fill(null));
    for (const { grid, x, y } of layers) {
        grid.forEach((row, gy) =>
            row.forEach((tone, gx) => {
                const px = x + gx;
                const py = y + gy;
                if (tone && py >= 0 && py < height && px >= 0 && px < width) {
                    (out[py] as (Tone | null)[])[px] = tone;
                }
            }),
        );
    }
    return out;
}

/** Trim transparent margin. */
export function crop(grid: Grid): Grid {
    let minX = Infinity;
    let maxX = -1;
    let minY = Infinity;
    let maxY = -1;
    grid.forEach((row, y) =>
        row.forEach((tone, x) => {
            if (!tone) return;
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }),
    );
    if (maxX < 0) return [];
    return grid.slice(minY, maxY + 1).map((row) => row.slice(minX, maxX + 1));
}

/** The canonical lockup: mark `m` in gold, the name in ink on its lower half, four pixels apart. */
export function lockup(): Grid {
    const mark = fromBlocks(MARK.m);
    const name = word();
    const markW = mark[0]?.length ?? 0;
    const width = markW + 4 + (name[0]?.length ?? 0);
    return compose(width, mark.length, [
        { grid: mark, x: 0, y: 0 },
        { grid: name, x: markW + 4, y: 5 },
    ]);
}

/** One terminal cell: a character plus tone roles for foreground and background. */
export type Cell = { ch: string; fg: Tone | null; bg: Tone | null };

/** Pixel grid → half-block cells, two pixel rows per cell row. An odd last row pads with transparent. */
export function toCells(grid: Grid): Cell[][] {
    const rows: Cell[][] = [];
    const width = grid[0]?.length ?? 0;
    for (let y = 0; y < grid.length; y += 2) {
        const top = grid[y] as (Tone | null)[];
        const bottom = grid[y + 1] ?? Array<Tone | null>(width).fill(null);
        const cells: Cell[] = [];
        for (let x = 0; x < width; x++) {
            const t = top[x] ?? null;
            const b = bottom[x] ?? null;
            if (!t && !b) cells.push({ ch: " ", fg: null, bg: null });
            else if (t && b)
                cells.push(t === b ? { ch: "█", fg: t, bg: null } : { ch: "▀", fg: t, bg: b });
            else if (t) cells.push({ ch: "▀", fg: t, bg: null });
            else cells.push({ ch: "▄", fg: b, bg: null });
        }
        rows.push(cells);
    }
    return rows;
}

/** Cells → plain half-block text, no color, trailing spaces trimmed. */
export function toText(cells: Cell[][]): string {
    return cells
        .map((row) =>
            row
                .map((c) => c.ch)
                .join("")
                .replace(/\s+$/, ""),
        )
        .join("\n");
}

/** Cells → HTML spans. Rows keep their full width so a frame never re-centers as it animates. */
export function toHtml(cells: Cell[][], palette: Palette): string {
    return cells
        .map((row) =>
            row
                .map((c) => {
                    if (!c.fg) return " ";
                    const bg = c.bg ? `;background:${palette[c.bg]}` : "";
                    return `<span style="color:${palette[c.fg]}${bg}">${c.ch}</span>`;
                })
                .join(""),
        )
        .join("\n");
}

const hexToRgb = (hex: string): [number, number, number] => [
    Number.parseInt(hex.slice(1, 3), 16),
    Number.parseInt(hex.slice(3, 5), 16),
    Number.parseInt(hex.slice(5, 7), 16),
];

/** Cells → one line per row with 24-bit ANSI color, reset at each row's end. */
export function toAnsi(cells: Cell[][], palette: Palette = DARK): string {
    const fg = (hex: string) => `\x1b[38;2;${hexToRgb(hex).join(";")}m`;
    const bg = (hex: string) => `\x1b[48;2;${hexToRgb(hex).join(";")}m`;
    return cells
        .map((row) => {
            let line = "";
            let open = false;
            for (const c of row) {
                if (!c.fg) {
                    if (open) line += "\x1b[0m";
                    open = false;
                    line += " ";
                    continue;
                }
                line += fg(palette[c.fg]) + (c.bg ? bg(palette[c.bg]) : "") + c.ch;
                open = true;
                if (c.bg) {
                    line += "\x1b[0m";
                    open = false;
                }
            }
            return line.replace(/\s+$/, "") + (open ? "\x1b[0m" : "");
        })
        .join("\n");
}

/** Pixel grid → SVG of square rects, one per pixel, crisp edges. Transparent ground. */
export function toSvg(grid: Grid, palette: Palette, scale: number = 1): string {
    const height = grid.length;
    const width = grid[0]?.length ?? 0;
    const rects = grid
        .flatMap((row, y) =>
            row.map((tone, x) =>
                tone
                    ? `<rect x="${x * scale}" y="${y * scale}" width="${scale}" height="${scale}" fill="${palette[tone]}"/>`
                    : "",
            ),
        )
        .filter(Boolean)
        .join("");
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width * scale} ${height * scale}" width="${width * scale}" height="${height * scale}" shape-rendering="crispEdges">${rects}</svg>`;
}

// --- splash ---------------------------------------------------------------------------------
//
// Doubling: pixels switch on in place in ordered-dither (Bayer) sequence, in beats that shorten.
// Each pixel lands dim and goes full a tick later. On the hit the name types one letter per
// tick; the cursor stays three ticks after the last letter and goes out. Thirty ticks a second.
// Every frame is a grid, so the same function drives the page and an ANSI terminal.

export const TICK_MS = 1000 / 30;
const ANTICIPATION = 3;
const BEATS = [2, 2, 2, 1, 1, 1, 1] as const;
const COUNTS = [3, 6, 10, 14, 18, 22] as const;
const CURSOR_TICKS = 3;
const BAYER = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
] as const;

/** Tick of the hit: the last beat, when the remaining pixels land and the name starts. */
export const HIT_TICK = ANTICIPATION + BEATS.reduce((a, b) => a + b, 0);

/** Last tick with any change; the frame after it is the lockup. */
export const END_TICK = HIT_TICK + NAME.length + CURSOR_TICKS;

type Point = { x: number; y: number };

const beatStart = (k: number): number =>
    ANTICIPATION + BEATS.slice(0, k).reduce<number>((a, b) => a + b, 0);

/** Deterministic tiebreak within a Bayer level, so the order is stable across runs. */
function lcg(seed: number): () => number {
    let s = seed;
    return () => {
        s = (s * 16807) % 2147483647;
        return s / 2147483647;
    };
}

function landing(grid: Grid): Map<Point, number> {
    const pixels: Point[] = [];
    grid.forEach((row, y) => row.forEach((tone, x) => tone && pixels.push({ x, y })));
    const rnd = lcg(5);
    const jitter = new Map(pixels.map((p) => [p, rnd()]));
    const key = (p: Point) => (BAYER[p.y % 4] as readonly number[])[p.x % 4] as number;
    pixels.sort((a, b) => key(a) - key(b) || (jitter.get(a) ?? 0) - (jitter.get(b) ?? 0));
    const ticks = new Map<Point, number>();
    let i = 0;
    COUNTS.forEach((n, k) => {
        for (let c = 0; c < n && i < pixels.length; c++, i++)
            ticks.set(pixels[i] as Point, beatStart(k));
    });
    for (; i < pixels.length; i++) ticks.set(pixels[i] as Point, HIT_TICK);
    return ticks;
}

const MARK_GRID = fromBlocks(MARK.m);
const LANDING = landing(MARK_GRID);
const MARK_W = MARK_GRID[0]?.length ?? 0;
const NAME_X = MARK_W + 4;
const NAME_Y = 5;
const GLYPHS = [...NAME].map((ch) => word(ch));

/** The splash at a tick, as the lockup-sized grid. Ticks past `END_TICK` return the lockup. */
export function splashFrame(tick: number): Grid {
    const name = word();
    const width = NAME_X + (name[0]?.length ?? 0);
    const out: Grid = Array.from({ length: MARK_GRID.length }, () =>
        Array<Tone | null>(width).fill(null),
    );
    if (tick < ANTICIPATION) {
        if (tick >= 1) (out[7] as (Tone | null)[])[5] = tick === 1 ? "dim" : "gold";
        return out;
    }
    for (const [p, land] of LANDING) {
        if (tick >= land) (out[p.y] as (Tone | null)[])[p.x] = tick - land < 1 ? "dim" : "gold";
    }
    const typed = Math.max(0, Math.min(NAME.length, tick - HIT_TICK));
    let x = NAME_X;
    for (let i = 0; i < typed; i++) {
        const glyph = GLYPHS[i] as Grid;
        glyph.forEach((row, gy) =>
            row.forEach((tone, gx) => {
                if (tone) (out[NAME_Y + gy] as (Tone | null)[])[x + gx] = "ink";
            }),
        );
        x += (glyph[0]?.length ?? 0) + 1;
    }
    if (tick >= HIT_TICK && tick < END_TICK) {
        for (let y = 2; y < 7; y++)
            for (let k = 0; k < 4; k++) (out[NAME_Y + y] as (Tone | null)[])[x + k] = "gold";
    }
    return out;
}

/** Every splash frame as ANSI lines, ready to write with cursor-up between them. */
export function splashAnsi(palette: Palette = DARK): string[] {
    const frames: string[] = [];
    for (let t = 0; t <= END_TICK; t++) frames.push(toAnsi(toCells(splashFrame(t)), palette));
    return frames;
}

/**
 * Drives the splash into `el` at thirty ticks a second: `render` turns each frame's grid into
 * markup. Ticks are quantized to the clock, so a fast display shows no extra frames. Honors
 * reduced motion by rendering the resting lockup once. Returns a replay function.
 */
export function runSplash(
    el: Element,
    render: (grid: Grid) => string,
    reduced: boolean = false,
): () => void {
    let start = 0;
    let last = -1;
    let raf = 0;
    const frame = () => {
        const tick = reduced ? END_TICK + 1 : Math.floor((performance.now() - start) / TICK_MS);
        if (tick !== last) {
            last = tick;
            el.innerHTML = render(splashFrame(Math.min(tick, END_TICK + 1)));
        }
        if (tick <= END_TICK) raf = requestAnimationFrame(frame);
    };
    const replay = () => {
        cancelAnimationFrame(raf);
        start = performance.now();
        last = -1;
        frame();
    };
    replay();
    return replay;
}
