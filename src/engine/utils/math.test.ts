import { expect, test } from "bun:test";
import {
    composeMat4,
    eulerToQuat,
    invertMat4,
    lookAt,
    lookAtRotation,
    lookAtRotationInto,
    multiplyMat4,
    perspective,
    quatToEuler,
    rotateQuatByEuler,
} from "./math";

// `lookAtRotation` and `lookAt` are two readings of one orientation: `lookAtRotation` returns it as a quaternion an entity is
// posed with, `lookAt` as the view matrix a projection multiplies. StandardRenderer poses each shadow light camera with
// `lookAtRotation` and renders that light through `lookAt`, so the pack's cull frustum is the render's frustum only
// while `invertMat4(composeMat4(eye, lookAtRotation(...)))` is `lookAt(...)`. Nothing else in the tree reads that agreement.

// eye/target pairs whose direction is not parallel to the up vector each row passes. A direction that is
// parallel is degenerate for both functions, and they resolve it differently — `lookAtRotation` nudges the up vector,
// `lookAt` substitutes an axis — so the two frames differ by a roll there. StandardRenderer never poses one: a cascade's
// up comes from the sun's own snap-plane basis, and a cube face's from its face table.
const CASES: { eye: [number, number, number]; target: [number, number, number] }[] = [
    { eye: [0, 10, 0.001], target: [0, 0, 0] },
    { eye: [5, 3, -2], target: [-1, 0.5, 4] },
    { eye: [-120, 60, 80], target: [0, 0, 0] },
    { eye: [0.25, 8.5, 0.75], target: [3.25, -3, -2.5] },
];

// the view matrix `lookAtRotation`'s quaternion implies: pose an entity at `eye` with it, then invert its world matrix
function viewFromAim(
    eye: [number, number, number],
    q: [number, number, number, number],
): Float32Array {
    const world = composeMat4(eye[0], eye[1], eye[2], ...q, 1, 1, 1);
    return invertMat4(world);
}

test("infinite reverse-Z perspective keeps depth precision beyond the culling far plane", () => {
    const projection = perspective(60, 1, 0.1);
    const depthAt = (distance: number) =>
        (projection[10]! * -distance + projection[14]!) / distance;
    expect(depthAt(0.1)).toBeCloseTo(1, 6);
    expect(depthAt(1_000_000)).toBeGreaterThan(0);
    expect(depthAt(1_000_000)).toBeLessThan(depthAt(1000));
});

test("a camera posed with lookAtRotation implies the view matrix lookAt builds for the same eye and target", () => {
    for (const { eye, target } of CASES) {
        const q = lookAtRotation(eye[0], eye[1], eye[2], target[0], target[1], target[2]);
        const view = viewFromAim(eye, q);
        const expected = lookAt(eye[0], eye[1], eye[2], target[0], target[1], target[2]);
        for (let i = 0; i < 16; i++) expect(view[i]).toBeCloseTo(expected[i], 4);
    }
});

test("lookAtRotation and lookAt agree for a caller-supplied up vector", () => {
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
            const q = lookAtRotation(
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
});

test("the view matrix a camera posed with lookAtRotation implies places the eye at the view origin", () => {
    for (const { eye, target } of CASES) {
        const q = lookAtRotation(eye[0], eye[1], eye[2], target[0], target[1], target[2]);
        const view = viewFromAim(eye, q);
        // the eye in view space: view · [eye, 1], read from the translation the multiply produces
        const world = composeMat4(eye[0], eye[1], eye[2], 0, 0, 0, 1, 1, 1, 1);
        const composed = multiplyMat4(view, world);
        expect(composed[12]).toBeCloseTo(0, 4);
        expect(composed[13]).toBeCloseTo(0, 4);
        expect(composed[14]).toBeCloseTo(0, 4);
    }
});

// The expected values are three.js's own output for radians, intrinsic XYZ order
// (reference/three.js 1e7e3cff30, `new Quaternion().setFromEuler(new Euler(x, y, z))`, `Euler.setFromQuaternion` and
// `Quaternion.multiply`), not this file's formula.
const THREE_EULER: { euler: [number, number, number]; quat: [number, number, number, number] }[] = [
    {
        euler: [1.0, 0.9, 2.1],
        quat: [0.5459113007037496, -0.18453238783560397, 0.789212571726169, 0.21230206365694462],
    },
    {
        euler: [-0.4, 0.3, 1.2],
        quat: [-0.07943052840097997, 0.23179560612167038, 0.5226700723191405, 0.8165645006014729],
    },
];

function expectClose(actual: readonly number[], expected: readonly number[]): void {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) expect(actual[i]).toBeCloseTo(expected[i], 12);
}

test("lookAtRotationInto matches the fresh quaternion, supports aliased inputs and overwrites reused output", () => {
    const out = new Float64Array(4);
    for (const eye of [
        [2, 3, -4],
        [0, 1, 0],
        [0, 0, 0],
    ]) {
        for (const up of [
            [0, 1, 0],
            [1, 0, 0],
        ]) {
            const expected = lookAtRotation(
                eye[0]!,
                eye[1]!,
                eye[2]!,
                0,
                0,
                0,
                up[0],
                up[1],
                up[2],
            );
            out.fill(99);
            const aim = new Float64Array([...eye, 0, 0, 0, ...up]);
            lookAtRotationInto(aim, out);
            expectClose(Array.from(out), expected);
            lookAtRotationInto(aim, aim);
            expectClose(Array.from(aim.subarray(0, 4)), expected);
            const retained: typeof expected = [...expected];
            lookAtRotation(1, 2, 3, 4, 5, 6);
            expect(expected).toEqual(retained);
        }
    }
    expect(() => lookAtRotationInto(new Float64Array([NaN, 0, 0, 0, 0, 0, 0, 1, 0]), out)).toThrow(
        "non-finite",
    );
});

test("eulerToQuat takes radians in XYZ order, as three.js setFromEuler", () => {
    for (const { euler, quat } of THREE_EULER) expectClose(eulerToQuat(...euler), quat);
});

test("quatToEuler returns radians in XYZ order, as three.js setFromQuaternion", () => {
    for (const { euler, quat } of THREE_EULER) expectClose(quatToEuler(...quat), euler);
});

test("quatToEuler at the y = pi/2 pole folds the rotation into x, as three.js does", () => {
    // three.js: setFromEuler(new Euler(0.7, Math.PI / 2, 0)), then setFromQuaternion
    const pole: [number, number, number, number] = [
        0.24246536490574874, 0.664236815315985, 0.24246536490574871, 0.6642368153159851,
    ];
    const [x, y, z] = quatToEuler(...pole);
    expect(x).toBeCloseTo(0.7, 12);
    expect(y).toBeCloseTo(1.5707963118937354, 12);
    expect(z).toBe(0);
});

test("rotateQuatByEuler applies a radian delta in the quaternion's local frame, as three.js q.multiply", () => {
    const [a, b] = THREE_EULER;
    expectClose(rotateQuatByEuler(0, 0, 0, 1, ...b.euler), b.quat);
    // three.js: setFromEuler(new Euler(1.0, 0.9, 2.1)).multiply(setFromEuler(new Euler(-0.4, 0.3, 1.2)))
    expectClose(
        rotateQuatByEuler(...a.quat, ...b.euler),
        [0.14952296061759973, -0.4494909822031051, 0.8672892402274799, -0.15300364359019847],
    );
});
