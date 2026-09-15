import { expect } from "bun:test";
import { check } from "../../harness/check";
import { aim, compose, invert, lookAt, multiply } from "./math";

// `aim` and `lookAt` are two readings of one orientation: `aim` returns it as a quaternion an entity is
// posed with, `lookAt` as the view matrix a projection multiplies. Sear poses each shadow light camera with
// `aim` and renders that light through `lookAt`, so the pack's cull frustum is the render's frustum only
// while `invert(compose(eye, aim(...)))` is `lookAt(...)`. Nothing else in the tree reads that agreement.

// eye/target pairs whose direction is not parallel to the up vector each row passes. A direction that is
// parallel is degenerate for both functions, and they resolve it differently — `aim` nudges the up vector,
// `lookAt` substitutes an axis — so the two frames differ by a roll there. Sear never poses one: a cascade's
// up comes from the sun's own snap-plane basis, and a cube face's from its face table.
const CASES: { eye: [number, number, number]; target: [number, number, number] }[] = [
    { eye: [0, 10, 0.001], target: [0, 0, 0] },
    { eye: [5, 3, -2], target: [-1, 0.5, 4] },
    { eye: [-120, 60, 80], target: [0, 0, 0] },
    { eye: [0.25, 8.5, 0.75], target: [3.25, -3, -2.5] },
];

// the view matrix `aim`'s quaternion implies: pose an entity at `eye` with it, then invert its world matrix
function viewFromAim(
    eye: [number, number, number],
    q: { x: number; y: number; z: number; w: number },
): Float32Array {
    const world = compose(eye[0], eye[1], eye[2], q.x, q.y, q.z, q.w, 1, 1, 1);
    return invert(world);
}

check(
    "aim and lookAt agree on the default up",
    {
        claim: "the quaternion aim returns orients a camera differently from the view matrix lookAt builds for the same eye and target, so a shadow light would cull against one frustum and render through another",
        subject: ["src/engine/utils/math.ts"],
    },
    () => {
        for (const { eye, target } of CASES) {
            const q = aim(eye[0], eye[1], eye[2], target[0], target[1], target[2]);
            const view = viewFromAim(eye, q);
            const expected = lookAt(eye[0], eye[1], eye[2], target[0], target[1], target[2]);
            for (let i = 0; i < 16; i++) expect(view[i]).toBeCloseTo(expected[i], 4);
        }
    },
);

check(
    "aim and lookAt agree on an explicit up",
    {
        claim: "aim and lookAt disagree once a caller supplies its own up vector, so a cascade or cube face posed with a non-default up would cull against a rolled frustum",
        subject: ["src/engine/utils/math.ts"],
    },
    () => {
        // the ups sear passes: a cascade's snap-plane basis, and a cube face's +Z/-Z hint
        const ups: [number, number, number][] = [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ];
        // a straight-down view, the pose the sun takes over flat ground: degenerate against the default up,
        // which is why the fit hands it the perpendicular basis instead
        const cases: typeof CASES = [...CASES, { eye: [0.25, 50, 0.75], target: [0.25, 0, 0.75] }];
        for (const { eye, target } of cases) {
            for (const up of ups) {
                // skip the pair that is parallel to this up — the degenerate case above
                const dx = eye[0] - target[0];
                const dy = eye[1] - target[1];
                const dz = eye[2] - target[2];
                const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
                const dot = Math.abs((dx * up[0] + dy * up[1] + dz * up[2]) / len);
                if (dot > 0.999) continue;
                const q = aim(
                    eye[0],
                    eye[1],
                    eye[2],
                    target[0],
                    target[1],
                    target[2],
                    up[0],
                    up[1],
                    up[2],
                );
                const view = viewFromAim(eye, q);
                const expected = lookAt(
                    eye[0],
                    eye[1],
                    eye[2],
                    target[0],
                    target[1],
                    target[2],
                    up[0],
                    up[1],
                    up[2],
                );
                for (let i = 0; i < 16; i++) expect(view[i]).toBeCloseTo(expected[i], 4);
            }
        }
    },
);

check(
    "an aimed camera's view matrix carries the eye to the origin",
    {
        claim: "the view matrix an aimed camera implies does not place the eye at the view origin, so every shadow cast from it would be offset from the light",
        subject: ["src/engine/utils/math.ts"],
    },
    () => {
        for (const { eye, target } of CASES) {
            const q = aim(eye[0], eye[1], eye[2], target[0], target[1], target[2]);
            const view = viewFromAim(eye, q);
            // the eye in view space: view · [eye, 1], read from the translation the multiply produces
            const world = compose(eye[0], eye[1], eye[2], 0, 0, 0, 1, 1, 1, 1);
            const composed = multiply(view, world);
            expect(composed[12]).toBeCloseTo(0, 4);
            expect(composed[13]).toBeCloseTo(0, 4);
            expect(composed[14]).toBeCloseTo(0, 4);
        }
    },
);
