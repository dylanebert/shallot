import { readFileSync } from "node:fs";
import { PNG } from "pngjs";
import { skipReason } from "../../../../scripts/verify";

// Shared seam for the four relocated engine-flow tiers. Each tier boots one ejected fixture app under
// `flows/` through the shipped `shallot verify` and asserts what the flow always asserted; only the
// paths changed. Two things live here because more than one tier needs them: the display refusal, and
// the node-side pixel reads that moved out of the retired `scripts/flows.ts` (paint containment is not
// observable from inside the page, so the screenshot is read here).

/** Refuse rather than skip: a display gate with no display is an unavailable premise, not a pass. */
export function requireDisplay(): void {
    const reason = skipReason();
    if (reason)
        throw new Error(
            `flow tier needs the seat's own display and a conformant adapter (${reason}); run it headed on the seat`,
        );
}

/** The HUD fill the ui-containment fixture paints. Must match `lib.ts`'s inline `rgb(255,0,255)`. */
export const MAGENTA: [number, number, number] = [255, 0, 255];

const near = (a: number, b: number, t = 40): boolean => Math.abs(a - b) <= t;
export const isMagenta = (r: number, g: number, b: number): boolean =>
    near(r, MAGENTA[0]) && near(g, MAGENTA[1]) && near(b, MAGENTA[2]);

/** Read a captured frame as `(x, y) -> rgb`, plus its dimensions. */
export function frame(path: string): {
    width: number;
    height: number;
    at: (x: number, y: number) => [number, number, number];
} {
    const { width, height, data } = PNG.sync.read(readFileSync(path));
    return {
        width,
        height,
        at: (x, y) => {
            const i = (y * width + x) * 4;
            return [data[i], data[i + 1], data[i + 2]];
        },
    };
}

/** The chrome sample points: well inside the fixture's 64px border, plus the strip directly above the
 *  canvas. Every one of them is host chrome the contained UI must never paint. */
export function chromePoints(width: number, height: number): [number, number][] {
    return [
        [8, 8],
        [width - 9, 8],
        [8, height - 9],
        [width - 9, height - 9],
        [Math.floor(width / 2), 8],
    ];
}
