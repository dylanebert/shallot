import { expect, test } from "bun:test";
import gold from "../../../../crates/physics/tests/math.gold.json";
import type { Mat3, Quat, Vec3 } from "../common/records";
import { kernel } from "./kernel";
import { matrix, quaternion, vector } from "./math";

const v = (a: number[], i = 0): Vec3 => ({ x: a[i], y: a[i + 1], z: a[i + 2] });
const q = (a: number[], i = 0): Quat => ({ v: v(a, i), s: a[i + 3] });
const m = (a: number[], i = 0): Mat3 => ({ cx: v(a, i), cy: v(a, i + 3), cz: v(a, i + 6) });
const fv = (a: Vec3) => [a.x, a.y, a.z];
const fq = (a: Quat) => [...fv(a.v), a.s];
const fm = (a: Mat3) => [...fv(a.cx), ...fv(a.cy), ...fv(a.cz)];

// Cargo math_gold owns the full scalar corpus; these rows also cross the WASM reader boundary.
const readers: Record<string, (a: number[]) => number[]> = {
    normalizeVec3: (a) => fv(vector.normalize(v(a))),
    normalizeQuat: (a) => fq(quaternion.normalize(q(a))),
    makeQuatFromAxisAngle: (a) => fq(quaternion.fromAxisAngle(v(a), a[3])),
    makeMatrixFromQuat: (a) => fm(matrix.fromQuat(q(a))),
    mulMM: (a) => fm(matrix.mul(m(a), m(a, 9))),
    transformPoint: (a) => {
        const k = kernel(undefined);
        new Float32Array(k.memory.buffer, k.mathInputPtr(), a.length).set(a);
        return Array.from(new Float32Array(k.memory.buffer, k.mathRead(14), 3));
    },
};
const decode = (hex: string) =>
    new Float32Array(new Uint32Array([Number.parseInt(hex, 16)]).buffer)[0];
const bits = (a: number) => new Uint32Array(new Float32Array([a]).buffer)[0];
for (const row of gold.cases) {
    const reader = readers[row.fn];
    if (!reader) continue;
    test(`kernel math reader ${row.fn} preserves native Box3D bits`, () => {
        const actual = reader(row.in.map(decode));
        expect(actual.map(bits)).toEqual(row.out.map((hex) => Number.parseInt(hex, 16)));
    });
}
