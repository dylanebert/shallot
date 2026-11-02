import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { type App, createApp } from "../../engine";
import { hull2D, type Point2D, simplifyHull2D } from "./hull2d";
import { kernel } from "./kernel/kernel";
import { StandardPhysicsPlugin } from "./runtime";

type Row = [number, number, number, number];
type Case = [number, Row[], Row[], Row[]];
const points = (rows: Row[]): Point2D[] =>
    rows.map(([x, y, separation, originalIndex]) => ({
        p: { x: Math.fround(x), y: Math.fround(y) },
        separation: Math.fround(separation),
        originalIndex,
    }));
setDefaultTimeout(TIMEOUT.node);
let app: App;
let cases: Case[];
beforeAll(async () => {
    app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    cases = JSON.parse(await Bun.file(new URL("./hull2d.gold.json", import.meta.url)).text());
});
afterAll(() => app.dispose());
test("public 2D hull construction and simplification equal all native point and metadata records", () => {
    for (const [i, [target, input, hull, simplified]] of cases.entries()) {
        const source = points(input);
        const original = structuredClone(source);
        const computed = hull2D(app.world, source);
        expect(computed, `case ${i} hull`).toEqual(points(hull));
        expect(source, `case ${i} input`).toEqual(original);
        const reduced = simplifyHull2D(app.world, computed, target);
        expect(reduced, `case ${i} simplified`).toEqual(points(simplified));
        expect(computed, `case ${i} retained hull`).toEqual(points(hull));
    }
});
test("2D hull records belong to the caller across other worlds and kernel memory growth", async () => {
    const other = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const input = points(cases.at(-1)![1]);
        const result = hull2D(app.world, input);
        const expected = structuredClone(result);
        hull2D(other.world, points(cases[2]![1]));
        kernel(app.world).hullBuildInputPtr(100_000);
        expect(hull2D(app.world, input)).toEqual(expected);
        expect(result).toEqual(expected);
    } finally {
        other.dispose();
    }
});
test("2D hull authoring rejects non-f32 coordinates, non-int32 identity and invalid simplification targets", () => {
    const point: Point2D = { p: { x: 0, y: 0 }, separation: 0, originalIndex: 0 };
    for (const x of [NaN, Infinity, Number.MAX_VALUE])
        expect(() => hull2D(app.world, [{ ...point, p: { x, y: 0 } }])).toThrow("finite f32");
    for (const originalIndex of [0.5, 2147483648, -2147483649])
        expect(() => hull2D(app.world, [{ ...point, originalIndex }])).toThrow(
            "int32 originalIndex",
        );
    for (const target of [2, 3.5, Infinity, 2147483648])
        expect(() => simplifyHull2D(app.world, [], target)).toThrow("target must be an integer");
    expect(
        simplifyHull2D(
            app.world,
            points([
                [0, 0, 0, 0],
                [1, 0, 1, 1],
                [2, 0, 2, 2],
                [2, 1, 3, 3],
                [0, 1, 4, 4],
            ]),
            6,
        ).map((p) => p.originalIndex),
    ).toEqual([0, 2, 3, 4]);
});
