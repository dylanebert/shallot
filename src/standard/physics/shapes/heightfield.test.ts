import { expect } from "bun:test";
import { check } from "../../../harness/check";
import {
    type CastOutput,
    emptyCastOutput,
    type RayCastInput,
    type ShapeCastInput,
    type ShapeCastPairInput,
    type ShapeProxy,
    shapeCast,
} from "../collision/distance";
import { aabb, intersectRayTriangle, type Vec3, xf } from "../common/math";
import gold from "./geometry.gold.json";
import {
    createGrid,
    createHeightField,
    createWave,
    getHeightFieldMaterial,
    getHeightFieldTriangle,
    HEIGHT_FIELD_HOLE,
    type HeightFieldData,
    overlapHeightField,
    rayCastHeightField,
    shapeCastHeightField,
} from "./heightfield";

const dv = new DataView(new ArrayBuffer(4));
function fromBits(hex: string): number {
    dv.setUint32(0, Number.parseInt(hex, 16));
    return dv.getFloat32(0);
}
function bits(f: number): string {
    dv.setFloat32(0, f);
    return dv.getUint32(0).toString(16).padStart(8, "0");
}
function bitEqual(got: number, want: string, label: string) {
    const w = fromBits(want);
    if (!Object.is(got, w)) {
        throw new Error(`${label}: got 0x${bits(got)} (${got}), want ${want} (${w})`);
    }
}
function vecEqual(got: Vec3, want: string[], label: string) {
    bitEqual(got.x, want[0], `${label}.x`);
    bitEqual(got.y, want[1], `${label}.y`);
    bitEqual(got.z, want[2], `${label}.z`);
}

type TriangleGold = {
    index: number;
    i1: number;
    i2: number;
    i3: number;
    flags: number;
    vertices: string[][];
};
type HeightFieldGold = {
    name: string;
    columnCount: number;
    rowCount: number;
    clockwise: boolean;
    minHeight: string;
    maxHeight: string;
    heightScale: string;
    scale: string[];
    boundsLower: string[];
    boundsUpper: string[];
    compressedHeights: number[];
    materialIndices: number[];
    flags: number[];
    triangles: TriangleGold[];
};

const hfGold = (name: string) =>
    gold.heightFields.find((h) => h.name === name) as unknown as HeightFieldGold;

// The bumpy field's authored heights (mirrors s_bump in fixtures/geometry_gold.c): all f32-exact.
const bumpHeights = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0];

function assertHeightField(hf: HeightFieldData, g: HeightFieldGold) {
    expect(hf.columnCount, `${g.name} columnCount`).toBe(g.columnCount);
    expect(hf.rowCount, `${g.name} rowCount`).toBe(g.rowCount);
    expect(hf.clockwise, `${g.name} clockwise`).toBe(g.clockwise);
    bitEqual(hf.minHeight, g.minHeight, `${g.name} minHeight`);
    bitEqual(hf.maxHeight, g.maxHeight, `${g.name} maxHeight`);
    bitEqual(hf.heightScale, g.heightScale, `${g.name} heightScale`);
    vecEqual(hf.scale, g.scale, `${g.name} scale`);
    vecEqual(hf.aabb.lowerBound, g.boundsLower, `${g.name} boundsLower`);
    vecEqual(hf.aabb.upperBound, g.boundsUpper, `${g.name} boundsUpper`);

    expect(hf.compressedHeights, `${g.name} compressedHeights`).toEqual(g.compressedHeights);
    expect(hf.materialIndices, `${g.name} materialIndices`).toEqual(g.materialIndices);
    expect(hf.flags, `${g.name} flags`).toEqual(g.flags);

    // Every non-hole triangle: decompressed vertices + indices + winding-remapped flags.
    for (const tg of g.triangles) {
        expect(getHeightFieldMaterial(hf, tg.index), `${g.name} tri${tg.index} material`).not.toBe(
            HEIGHT_FIELD_HOLE,
        );
        const t = getHeightFieldTriangle(hf, tg.index);
        expect(t.i1, `${g.name} tri${tg.index}.i1`).toBe(tg.i1);
        expect(t.i2, `${g.name} tri${tg.index}.i2`).toBe(tg.i2);
        expect(t.i3, `${g.name} tri${tg.index}.i3`).toBe(tg.i3);
        expect(t.flags, `${g.name} tri${tg.index}.flags`).toBe(tg.flags);
        vecEqual(t.vertices[0], tg.vertices[0], `${g.name} tri${tg.index}.v0`);
        vecEqual(t.vertices[1], tg.vertices[1], `${g.name} tri${tg.index}.v1`);
        vecEqual(t.vertices[2], tg.vertices[2], `${g.name} tri${tg.index}.v2`);
    }
}

const bumpDef = (clockwiseWinding: boolean) => ({
    heights: bumpHeights,
    materialIndices: null,
    scale: { x: 1, y: 0.5, z: 2 },
    countX: 5,
    countZ: 5,
    globalMinimumHeight: -256,
    globalMaximumHeight: 256,
    clockwiseWinding,
});

check(
    "height field build is bit-exact against the C geometry gold",
    {
        claim: "a height field built by createGrid or createHeightField would drift from the pinned C reference in its quantized heights, bounds, edge flags, winding remap or decompressed triangle vertices",
        tier: "step",
    },
    () => {
        const cases: Array<{ name: string; build: () => HeightFieldData }> = [
            { name: "grid", build: () => createGrid(5, 5, { x: 1, y: 1, z: 1 }, false) },
            { name: "grid-holes", build: () => createGrid(6, 6, { x: 1, y: 1, z: 1 }, true) },
            { name: "bump", build: () => createHeightField(bumpDef(false)) },
            { name: "bump-cw", build: () => createHeightField(bumpDef(true)) },
        ];
        for (const c of cases) {
            const g = hfGold(c.name);
            if (g === undefined) throw new Error(`height field gold case ${c.name}: missing`);
            assertHeightField(c.build(), g);
        }
    },
);

check(
    "height field wave authoring yields sound dimensions and finite triangles",
    {
        claim: "createWave would author a height field with the wrong row, column, material or flag counts, or a triangle decoding to a non-finite vertex or an invalid field AABB",
        tier: "step",
    },
    () => {
        // createWave uses Math.sin (not the portable b3 trig or C's sinf), so it is authoring-only and
        // can't be gold-tested; this guards its dimensions + that every triangle decodes validly.
        const hf = createWave(6, 8, { x: 1, y: 1, z: 1 }, 0.25, 0.5, false);
        expect(hf.rowCount, "wave rowCount").toBe(6);
        expect(hf.columnCount, "wave columnCount").toBe(8);
        expect(hf.compressedHeights.length, "wave compressedHeights.length").toBe(6 * 8);
        expect(hf.materialIndices.length, "wave materialIndices.length").toBe(5 * 7);
        expect(hf.flags.length, "wave flags.length").toBe(2 * 5 * 7);
        const triangleCount = 2 * 5 * 7;
        for (let i = 0; i < triangleCount; ++i) {
            const t = getHeightFieldTriangle(hf, i);
            for (const v of t.vertices) {
                expect(
                    Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z),
                    `wave tri${i} vertex finite`,
                ).toBe(true);
            }
        }
        expect(aabb.isValid(hf.aabb), "wave field AABB valid").toBe(true);
    },
);

// --- query behavioral checks (ported from test_height_field.c) -------------------------------

const near = (a: number, b: number, tol: number, label: string) => {
    if (!(Math.abs(a - b) <= tol)) {
        throw new Error(`${label}: got ${a}, want ${b} within ${tol}`);
    }
};

// Brute force: cast the proxy against every non-hole triangle and keep the closest hit. Ground truth
// for the grid-walked b3ShapeCastHeightField (getHeightFieldTriangle winds triangles the same way).
function bruteForceShapeCast(hf: HeightFieldData, input: ShapeCastInput): CastOutput {
    let best = emptyCastOutput();
    let bestFraction = input.maxFraction;
    const triangleCount = hf.flags.length;
    for (let t = 0; t < triangleCount; ++t) {
        if (hf.materialIndices[t >> 1] === HEIGHT_FIELD_HOLE) continue;
        const tri = getHeightFieldTriangle(hf, t);
        const pair: ShapeCastPairInput = {
            proxyA: { points: tri.vertices, count: 3, radius: 0 },
            proxyB: input.proxy,
            transform: xf.identity(),
            translationB: input.translation,
            maxFraction: bestFraction,
            canEncroach: input.canEncroach,
        };
        const out = shapeCast(pair);
        if (out.hit && out.fraction < bestFraction) {
            bestFraction = out.fraction;
            best = out;
            best.triangleIndex = t;
        }
    }
    return best;
}

function bruteForceRayCast(hf: HeightFieldData, input: RayCastInput): CastOutput {
    const best = emptyCastOutput();
    let bestFraction = input.maxFraction;
    const triangleCount = hf.flags.length;
    for (let t = 0; t < triangleCount; ++t) {
        if (hf.materialIndices[t >> 1] === HEIGHT_FIELD_HOLE) continue;
        const tri = getHeightFieldTriangle(hf, t);
        const alpha = intersectRayTriangle(
            input.origin,
            input.translation,
            tri.vertices[0],
            tri.vertices[1],
            tri.vertices[2],
        );
        if (alpha < bestFraction) {
            bestFraction = alpha;
            best.hit = true;
            best.fraction = alpha;
            best.triangleIndex = t;
        }
    }
    return best;
}

check(
    "height field ray cast lands on a flat surface with an up normal",
    {
        claim: "rayCastHeightField would miss a flat height field, or recover the wrong hit fraction or a surface normal that is not the field's up axis",
        tier: "step",
    },
    () => {
        // Tight quantization range keeps the recovered surface within ~1e-5 of y=0.
        const hf = createHeightField({
            heights: new Array(16).fill(0),
            materialIndices: new Array(9).fill(0),
            scale: { x: 1, y: 1, z: 1 },
            countX: 4,
            countZ: 4,
            globalMinimumHeight: -1,
            globalMaximumHeight: 1,
            clockwiseWinding: false,
        });
        const out = rayCastHeightField(hf, {
            origin: { x: 1.25, y: 10, z: 1.25 },
            translation: { x: 0, y: -20, z: 0 },
            maxFraction: 1,
        });
        expect(out.hit, "flat field ray hit").toBe(true);
        near(out.fraction, 0.5, 1e-5, "flat field ray fraction");
        near(out.normal.x, 0, 1e-5, "flat field ray normal.x");
        near(out.normal.y, 1, 1e-5, "flat field ray normal.y");
        near(out.normal.z, 0, 1e-5, "flat field ray normal.z");
    },
);

check(
    "height field overlap separates a proxy above the surface from one through it",
    {
        claim: "overlapHeightField would report a hit for a proxy hovering clear above the field, or miss one whose radius reaches the field surface",
        tier: "step",
    },
    () => {
        const hf = createGrid(4, 4, { x: 1, y: 1, z: 1 }, false);
        const cases: Array<{ name: string; proxy: ShapeProxy; want: boolean }> = [
            {
                name: "sphere above the surface",
                proxy: { points: [{ x: 1.5, y: 1, z: 1.5 }], count: 1, radius: 0.5 },
                want: false,
            },
            {
                name: "sphere through the surface",
                proxy: { points: [{ x: 1.5, y: 0, z: 1.5 }], count: 1, radius: 0.5 },
                want: true,
            },
        ];
        for (const c of cases) {
            expect(
                overlapHeightField(hf, xf.identity(), c.proxy),
                `height field overlap ${c.name}`,
            ).toBe(c.want);
        }
    },
);

check(
    "height field shape cast tests every cell the swept proxy straddles",
    {
        claim: "shapeCastHeightField would cull the one solid height field cell that sits on the trailing side of the swept proxy's x, z or corner boundary",
        tier: "step",
    },
    () => {
        // Only cell (0,0) is solid; the swept sphere's center is nudged just past each boundary so the
        // solid cell sits on the trailing side. A cull AABB pinned to the leading corner would miss it.
        const hf = createHeightField({
            heights: new Array(9).fill(0),
            materialIndices: [0, HEIGHT_FIELD_HOLE, HEIGHT_FIELD_HOLE, HEIGHT_FIELD_HOLE],
            scale: { x: 1, y: 1, z: 1 },
            countX: 3,
            countZ: 3,
            globalMinimumHeight: -1,
            globalMaximumHeight: 1,
            clockwiseWinding: false,
        });
        const radius = 0.3;
        const cast = (cx: number, cz: number) =>
            shapeCastHeightField(hf, {
                proxy: { points: [{ x: cx, y: 10, z: cz }], count: 1, radius },
                translation: { x: 0, y: -20, z: 0 },
                maxFraction: 1,
                canEncroach: false,
            });

        const cases = [
            { name: "past the x boundary", cx: 1.05, cz: 0.5, fraction: 0.4852098 },
            { name: "past the z boundary", cx: 0.5, cz: 1.05, fraction: 0.4852098 },
            { name: "past the corner", cx: 1.05, cz: 1.05, fraction: 0.4854226 },
        ];
        for (const c of cases) {
            const out = cast(c.cx, c.cz);
            expect(out.hit, `height field shape cast ${c.name}: hit`).toBe(true);
            near(out.fraction, c.fraction, 2e-3, `height field shape cast ${c.name}: fraction`);
        }
    },
);

const waveOrigins = (): Vec3[] => {
    const out: Vec3[] = [];
    for (let xi = 0; xi < 5; ++xi) {
        for (let zi = 0; zi < 5; ++zi) {
            out.push({ x: 1 + 4 * xi + 0.05, y: 4, z: 1 + 4 * zi + 0.05 });
        }
    }
    return out;
};

const label = (o: Vec3, d: Vec3) =>
    `origin (${o.x}, ${o.y}, ${o.z}) delta (${d.x}, ${d.y}, ${d.z})`;

check(
    "height field shape cast grid walk matches the brute-force cast",
    {
        claim: "the grid walk in shapeCastHeightField would disagree with a brute-force cast over every wave height field triangle on hit or fraction for some origin, sweep and proxy radius",
        tier: "step",
    },
    () => {
        const hf = createWave(10, 10, { x: 2, y: 1.5, z: 2 }, 0.1, 0.03333, false);
        const radii = [0.15, 0.4, 0.9];
        const deltas: Vec3[] = [
            { x: 0, y: -8, z: 0 },
            { x: 0, y: -8, z: 6.4 },
            { x: 5.1, y: -8, z: 0 },
            { x: 0, y: -8, z: -6.4 },
            { x: -5.1, y: -8, z: 0 },
            { x: 6, y: -8, z: 5 },
            { x: -7, y: -8, z: 4 },
            { x: 9, y: -3, z: -9 },
        ];
        const failures: string[] = [];
        for (const origin of waveOrigins()) {
            for (const delta of deltas) {
                for (const radius of radii) {
                    const input: ShapeCastInput = {
                        proxy: { points: [origin], count: 1, radius },
                        translation: delta,
                        maxFraction: 1,
                        canEncroach: false,
                    };
                    const grid = shapeCastHeightField(hf, input);
                    const brute = bruteForceShapeCast(hf, input);
                    const where = `${label(origin, delta)} radius ${radius}`;
                    if (grid.hit !== brute.hit) {
                        failures.push(`${where}: grid hit ${grid.hit}, brute hit ${brute.hit}`);
                    } else if (brute.hit && Math.abs(grid.fraction - brute.fraction) > 2e-3) {
                        failures.push(
                            `${where}: grid fraction ${grid.fraction}, brute ${brute.fraction}`,
                        );
                    }
                }
            }
        }
        expect(failures, "height field shape cast grid walk disagreements").toEqual([]);
    },
);

check(
    "height field ray cast grid walk matches the brute-force ray",
    {
        claim: "the grid walk in rayCastHeightField would disagree with a brute-force ray against every wave height field triangle on hit or fraction for some origin and translation",
        tier: "step",
    },
    () => {
        const hf = createWave(10, 10, { x: 2, y: 1.5, z: 2 }, 0.1, 0.03333, false);
        const deltas: Vec3[] = [
            { x: 0, y: -8, z: 0 },
            { x: 0, y: -8, z: 12 },
            { x: 12, y: -8, z: 0 },
            { x: 0, y: -8, z: -12 },
            { x: -12, y: -8, z: 0 },
            { x: 14, y: -8, z: 11 },
            { x: -13, y: -8, z: 9 },
            { x: 16, y: -4, z: -15 },
        ];
        const failures: string[] = [];
        for (const origin of waveOrigins()) {
            for (const delta of deltas) {
                const input: RayCastInput = { origin, translation: delta, maxFraction: 1 };
                const grid = rayCastHeightField(hf, input);
                const brute = bruteForceRayCast(hf, input);
                const where = label(origin, delta);
                if (grid.hit !== brute.hit) {
                    failures.push(`${where}: grid hit ${grid.hit}, brute hit ${brute.hit}`);
                } else if (brute.hit && Math.abs(grid.fraction - brute.fraction) > 1e-4) {
                    failures.push(
                        `${where}: grid fraction ${grid.fraction}, brute ${brute.fraction}`,
                    );
                }
            }
        }
        expect(failures, "height field ray cast grid walk disagreements").toEqual([]);
    },
);
