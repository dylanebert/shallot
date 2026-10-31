import { expect, test } from "bun:test";
import { World } from "../../src/engine";
import { shapeCast } from "../../src/standard/physics/collision/queries";
import { init } from "../../src/standard/physics/kernel/kernel";
import { nativeSseOutput } from "./native-evidence";

const world = new World();
await init(world, { threads: 0 });
const bits = new DataView(new ArrayBuffer(4));
const hex = (v: number) => {
    bits.setFloat32(0, v);
    return bits.getUint32(0).toString(16).padStart(8, "0");
};
const points = (x: number, lo: number, hi: number, z: number) => [
    { x, y: hi, z }, { x: -x, y: hi, z }, { x: -x, y: lo, z }, { x, y: lo, z },
    { x, y: hi, z: -z }, { x: -x, y: hi, z: -z }, { x: -x, y: lo, z: -z }, { x, y: lo, z: -z },
];

test("near-target box encroachment retains a rest gap and rejects receding, tangential, short and clipped sweeps", () => {
    const expected = nativeSseOutput("encroach-cast.c", "").trim().split("\n");
    const translations = [{ x: 1, y: 0.1, z: 0 }, { x: 1, y: 0, z: 0 }, { x: 1, y: -0.1, z: 0 }, { x: 0, y: -0.001, z: 0 }, { x: 0, y: -0.1, z: 0 }];
    const actual = translations.map((translationB, i) => {
        const out = shapeCast(world, {
            proxyA: { points: points(3, -0.5, 0, 3), count: 8, radius: 0 },
            proxyB: { points: points(0.5, -0.5, 0.5, 0.5), count: 8, radius: 0 },
            transform: { p: { x: 0, y: Math.fround(0.504), z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            translationB, maxFraction: i === 4 ? 0.001 : 1, canEncroach: true,
        });
        return [Number(out.hit), ...[out.fraction, out.point.x, out.point.y, out.point.z, out.normal.x, out.normal.y, out.normal.z].map(hex), out.iterations].join(" ");
    });
    expect(actual).toEqual(expected);
});
