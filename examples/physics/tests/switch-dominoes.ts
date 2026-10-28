import { test } from "bun:test";
import { expectSwitchesEqualFresh } from "./switch";

export const dominoSwitchRanges = {
    first: { start: 0, end: 150 },
    second: { start: 150, end: 300 },
    third: { start: 300, end: 450 },
    fourth: { start: 450, end: 600 },
};

export function testDominoSwitches(part: keyof typeof dominoSwitchRanges): void {
    const steps = dominoSwitchRanges[part];
    test(`switching Dominoes equals a fresh scene at steps ${steps.start}-${steps.end} after a changed, grabbed Bridge`, () =>
        expectSwitchesEqualFresh(["dominoes"], [], steps));
}
