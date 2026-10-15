import type { AABB, Mat3, Quat, Transform, Vec3 } from "../common/records";
import { kernel } from "./kernel";

// Parity inputs and expected intermediates must retain Box3D's arithmetic.
function read(operation: number, input: number[]): Float32Array {
    const k = kernel(undefined);
    new Float32Array(k.memory.buffer, k.mathInputPtr(), 18).set(input);
    return new Float32Array(k.memory.buffer, k.mathRead(operation), 9);
}
const v = (a: Vec3) => [a.x, a.y, a.z];
const q = (a: Quat) => [...v(a.v), a.s];
const m = (a: Mat3) => [...v(a.cx), ...v(a.cy), ...v(a.cz)];
function vectorResult(op: number, input: number[]): Vec3 {
    const a = read(op, input);
    return { x: a[0], y: a[1], z: a[2] };
}
function quaternionResult(op: number, input: number[]): Quat {
    const a = read(op, input);
    return { v: { x: a[0], y: a[1], z: a[2] }, s: a[3] };
}
function matrixResult(op: number, input: number[]): Mat3 {
    const a = read(op, input);
    return {
        cx: { x: a[0], y: a[1], z: a[2] },
        cy: { x: a[3], y: a[4], z: a[5] },
        cz: { x: a[6], y: a[7], z: a[8] },
    };
}
export const vector = {
    zero: (): Vec3 => ({ x: 0, y: 0, z: 0 }),
    axisX: (): Vec3 => ({ x: 1, y: 0, z: 0 }),
    axisZ: (): Vec3 => ({ x: 0, y: 0, z: 1 }),
    add: (a: Vec3, b: Vec3) => vectorResult(0, [...v(a), ...v(b)]),
    sub: (a: Vec3, b: Vec3) => vectorResult(1, [...v(a), ...v(b)]),
    scale: (s: number, a: Vec3) => vectorResult(2, [s, ...v(a)]),
    cross: (a: Vec3, b: Vec3) => vectorResult(3, [...v(a), ...v(b)]),
    dot: (a: Vec3, b: Vec3) => read(4, [...v(a), ...v(b)])[0],
    normalize: (a: Vec3) => vectorResult(5, v(a)),
    isNormalized: (a: Vec3) => read(22, v(a))[0] !== 0,
};
export const quaternion = {
    identity: (): Quat => ({ v: vector.zero(), s: 1 }),
    fromAxisAngle: (axis: Vec3, angle: number) => quaternionResult(6, [...v(axis), angle]),
    normalize: (a: Quat): Quat => {
        const k = kernel(undefined);
        const ptr = k.mathNormalizeQuat(a.v.x, a.v.y, a.v.z, a.s);
        const out = new Float32Array(k.memory.buffer, ptr, 4);
        return { v: { x: out[0], y: out[1], z: out[2] }, s: out[3] };
    },
    mul: (a: Quat, b: Quat) => quaternionResult(8, [...q(a), ...q(b)]),
    negate: (a: Quat) => quaternionResult(9, q(a)),
    conjugate: (a: Quat) => quaternionResult(10, q(a)),
    dot: (a: Quat, b: Quat) => read(11, [...q(a), ...q(b)])[0],
    rotate: (a: Quat, b: Vec3) => vectorResult(12, [...q(a), ...v(b)]),
    invRotate: (a: Quat, b: Vec3) => vectorResult(13, [...q(a), ...v(b)]),
};
export const transform = {
    identity: (): Transform => ({ p: vector.zero(), q: quaternion.identity() }),
};
export function transformWorldPoint(a: Transform, b: Vec3): Vec3 {
    const k = kernel(undefined);
    new Float32Array(k.memory.buffer, k.mathInputPtr(), 7).set([...q(a.q), ...v(b)]);
    const out = new Float32Array(k.memory.buffer, k.mathWorldPoint(a.p.x, a.p.y, a.p.z), 3);
    return { x: out[0], y: out[1], z: out[2] };
}
export const matrix = {
    zero: (): Mat3 => ({ cx: vector.zero(), cy: vector.zero(), cz: vector.zero() }),
    fromQuat: (a: Quat) => matrixResult(15, q(a)),
    mul: (a: Mat3, b: Mat3) => matrixResult(16, [...m(a), ...m(b)]),
    transpose: (a: Mat3) => matrixResult(17, m(a)),
    mulV: (a: Mat3, b: Vec3) => vectorResult(18, [...m(a), ...v(b)]),
};
export function intersectRayTriangle(start: Vec3, delta: Vec3, a: Vec3, b: Vec3, c: Vec3): number {
    return read(20, [...v(start), ...v(delta), ...v(a), ...v(b), ...v(c)])[0];
}
export const bounds = {
    contains: (a: AABB, b: AABB) =>
        read(21, [
            ...v(a.lowerBound),
            ...v(a.upperBound),
            ...v(b.lowerBound),
            ...v(b.upperBound),
        ])[0] !== 0,
};
