import { describe, expect, test } from "bun:test";
import { TRANSITION_TOLERANCE_PX } from "./capture";

describe("TRANSITION_TOLERANCE_PX", () => {
    test("is a small positive pixel bound, not a fraction or a huge slack", () => {
        // the mechanism argument (capture.ts's docstring) puts the formula's own band at ~0.5 px; the
        // constant should sit within a small integer multiple of that, not an order of magnitude off in
        // either direction (too tight would fail on ordinary filtering slop, too loose stops catching a
        // genuinely soft/misregistered edge — the exact defect this gate exists to find).
        expect(TRANSITION_TOLERANCE_PX).toBeGreaterThan(0);
        expect(TRANSITION_TOLERANCE_PX).toBeLessThanOrEqual(4);
    });
});
