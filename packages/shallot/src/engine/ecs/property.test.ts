import { beforeEach, describe, expect, test } from "bun:test";
import fc from "fast-check";
import { type Component, f32, State, sparse, vec2, vec4 } from "../..";
import { formatFields, parseFields } from "../scene/core";
import * as math from "../utils/math";
import { clear, register } from "./core";
import { schemas } from "./reflection";

// Synthetic components keep these property tests hermetic. The invariants under
// test — membership/query consistency and serialization round-trip across field
// kinds — are component-agnostic, so a real plugin's GPU-backed storage adds
// coupling (and a plugin-initialize dependency) without adding coverage.

const A = { value: [] as number[] };
const B = { value: [] as number[] };
const C = { value: [] as number[] };
const COMPONENTS: Component[] = [A, B, C];

function world(): State {
    clear();
    register("a", A);
    register("b", B);
    register("c", C);
    return new State();
}

type Command =
    | { type: "spawn" }
    | { type: "despawn"; entityIdx: number }
    | { type: "addComponent"; entityIdx: number; componentIdx: number }
    | { type: "removeComponent"; entityIdx: number; componentIdx: number };

const commandArb: fc.Arbitrary<Command> = fc.oneof(
    fc.constant({ type: "spawn" } as Command),
    fc.record({
        type: fc.constant("despawn" as const),
        entityIdx: fc.nat({ max: 49 }),
    }),
    fc.record({
        type: fc.constant("addComponent" as const),
        entityIdx: fc.nat({ max: 49 }),
        componentIdx: fc.nat({ max: COMPONENTS.length - 1 }),
    }),
    fc.record({
        type: fc.constant("removeComponent" as const),
        entityIdx: fc.nat({ max: 49 }),
        componentIdx: fc.nat({ max: COMPONENTS.length - 1 }),
    }),
);

describe("ECS command sequences", () => {
    test("invariants hold after arbitrary command sequences", () => {
        fc.assert(
            fc.property(fc.array(commandArb, { minLength: 5, maxLength: 30 }), (commands) => {
                const state = world();
                const alive: number[] = [];
                const despawned = new Set<number>();

                for (const cmd of commands) {
                    switch (cmd.type) {
                        case "spawn": {
                            const eid = state.create();
                            alive.push(eid);
                            despawned.delete(eid);
                            break;
                        }
                        case "despawn": {
                            if (alive.length === 0) break;
                            const idx = cmd.entityIdx % alive.length;
                            const eid = alive[idx];
                            state.destroy(eid);
                            alive.splice(idx, 1);
                            despawned.add(eid);
                            break;
                        }
                        case "addComponent": {
                            if (alive.length === 0) break;
                            const eid = alive[cmd.entityIdx % alive.length];
                            state.add(eid, COMPONENTS[cmd.componentIdx] as never);
                            break;
                        }
                        case "removeComponent": {
                            if (alive.length === 0) break;
                            const eid = alive[cmd.entityIdx % alive.length];
                            const component = COMPONENTS[cmd.componentIdx];
                            if (state.has(eid, component as never)) {
                                state.remove(eid, component as never);
                            }
                            break;
                        }
                    }

                    // Invariant: despawned entities absent
                    for (const eid of despawned) {
                        expect(state.exists(eid)).toBe(false);
                    }

                    // Invariant: query result matches has() for every alive entity
                    for (const component of COMPONENTS) {
                        const queryResult = new Set([...state.query([component as never])]);
                        for (const eid of alive) {
                            const has = state.has(eid, component as never);
                            expect(queryResult.has(eid)).toBe(has);
                        }
                    }
                }
            }),
            { numRuns: 100 },
        );
    });
});

describe("Serialization roundtrip", () => {
    // Typed synthetic components covering every codec field kind: vec4/vec2/float
    // via sparse storage, color + enum via traits.
    const Spatial = { pos: sparse(vec4), vel: sparse(vec2), speed: sparse(f32) };
    const Style = { tint: [] as number[], mode: [] as number[] };
    const formatHex = Object.assign((n: number) => "0x" + (n >>> 0).toString(16).padStart(6, "0"), {
        kind: "color" as const,
    });
    const Mode = { Off: 0, Low: 1, High: 2 } as const;

    beforeEach(() => {
        clear();
        register("spatial", Spatial, {
            defaults: () => ({ pos: [0, 0, 0, 0], vel: [0, 0], speed: 1 }),
        });
        register("style", Style, {
            defaults: () => ({ tint: 0xffffff, mode: 0 }),
            format: { tint: formatHex },
            parse: { tint: (v: string) => Number.parseInt(v.replace("0x", ""), 16) },
            enums: { mode: Mode },
        });
    });

    test("parseFields(formatFields(fields)) preserves values", () => {
        // formatFields serializes non-integers via toPrecision(7); the round-trip
        // relative error is bounded by a half-ULP at the 7th significant figure.
        const SerializeRel = 5e-7;
        const allSchemas = schemas();

        for (const s of allSchemas) {
            const arbs: Record<string, fc.Arbitrary<number>> = {};
            let generatable = 0;

            for (const field of s.fields) {
                switch (field.kind) {
                    case "float": {
                        if (field.fields) {
                            for (const f of field.fields) {
                                arbs[f] = fc.double({ min: -1000, max: 1000, noNaN: true });
                                generatable++;
                            }
                        } else {
                            arbs[field.name] = fc.double({ min: -1000, max: 1000, noNaN: true });
                            generatable++;
                        }
                        break;
                    }
                    case "vec2":
                    case "vec3":
                    case "vec4": {
                        if (field.fields) {
                            for (const f of field.fields) {
                                arbs[f] = fc.double({ min: -1000, max: 1000, noNaN: true });
                                generatable++;
                            }
                        }
                        break;
                    }
                    case "color": {
                        arbs[field.name] = fc.integer({ min: 0, max: 0xffffff });
                        generatable++;
                        break;
                    }
                    case "enum": {
                        if (field.options) {
                            arbs[field.name] = fc.constantFrom(...Object.values(field.options));
                            generatable++;
                        }
                        break;
                    }
                    // string: skip
                }
            }

            if (generatable === 0) continue;

            fc.assert(
                fc.property(fc.record(arbs), (fields) => {
                    const formatted = formatFields(s.name, fields as Record<string, number>);
                    if (!formatted) return;
                    const parsed = parseFields(s.name, formatted);

                    for (const key of Object.keys(parsed)) {
                        if (!(key in fields)) continue;
                        const a = fields[key];
                        const b = parsed[key];
                        if (typeof a === "number" && typeof b === "number") {
                            expect(Math.abs(b - a)).toBeLessThanOrEqual(
                                SerializeRel * Math.max(1, Math.abs(a)),
                            );
                        } else {
                            expect(b).toBe(a);
                        }
                    }
                }),
                { numRuns: 200 },
            );
        }
    });
});

describe("Math properties", () => {
    // f32 matrix-op error scales with element magnitude × the f32 unit roundoff
    // (2⁻²⁴). lookAt columns and M·M⁻¹≈I land on unit-magnitude targets; the
    // associativity / double-inverse products carry translations up to a few
    // hundred, so they accumulate proportionally more.
    const F32Unit = 1e-6;
    const F32Scaled = 1e-4;
    // f64 quaternion ops. slerp's near-parallel branch (dot > 0.9995) is an
    // unnormalized lerp; its chord deficit there is ≤ 1 − cos(½·acos(0.9995)) ≈
    // 1.3e-4. Endpoints return an input quaternion exactly, leaving only f64
    // norm noise. euler↔quat↔euler is exact in reals, amplified near gimbal by
    // 1/cos(89°) ≈ 57 over the safe domain.
    const SlerpNorm = 2e-4;
    const QuatExact = 1e-12;
    const EulerRt = 1e-9;

    const safeEulerArb = fc.record({
        x: fc.double({ min: -89, max: 89, noNaN: true }),
        y: fc.double({ min: -89, max: 89, noNaN: true }),
        z: fc.double({ min: -179, max: 179, noNaN: true }),
    });

    const unitQuatArb = fc
        .tuple(
            fc.double({ min: -1, max: 1, noNaN: true }),
            fc.double({ min: -1, max: 1, noNaN: true }),
            fc.double({ min: -1, max: 1, noNaN: true }),
            fc.double({ min: -1, max: 1, noNaN: true }),
        )
        .filter(([x, y, z, w]) => {
            const len = Math.sqrt(x * x + y * y + z * z + w * w);
            return len > 0.01;
        })
        .map(([x, y, z, w]) => {
            const len = Math.sqrt(x * x + y * y + z * z + w * w);
            return { x: x / len, y: y / len, z: z / len, w: w / len };
        });

    const posArb = fc.double({ min: -100, max: 100, noNaN: true, noDefaultInfinity: true });

    const rigidMatrixArb = fc
        .tuple(posArb, posArb, posArb, posArb, posArb, posArb)
        .filter(([ex, ey, ez, tx, ty, tz]) => {
            const dx = ex - tx,
                dy = ey - ty,
                dz = ez - tz;
            return dx * dx + dy * dy + dz * dz > 0.001;
        })
        .map(([ex, ey, ez, tx, ty, tz]) => math.lookAt(ex, ey, ez, tx, ty, tz));

    const perspectiveMatrixArb = fc
        .tuple(
            fc.double({ min: 10, max: 160, noNaN: true }),
            fc.double({ min: 0.1, max: 10, noNaN: true }),
            fc.double({ min: 0.01, max: 1, noNaN: true }),
            fc.double({ min: 10, max: 10000, noNaN: true }),
        )
        .map(([fov, aspect, near, far]) => math.perspective(fov, aspect, near, far));

    const tArb = fc.double({ min: 0, max: 1, noNaN: true });

    function quatDot(
        a: { x: number; y: number; z: number; w: number },
        b: { x: number; y: number; z: number; w: number },
    ): number {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    function quatNorm(q: { x: number; y: number; z: number; w: number }): number {
        return Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    }

    function matClose(a: Float32Array, b: Float32Array, eps: number): boolean {
        for (let i = 0; i < 16; i++) {
            if (Math.abs(a[i] - b[i]) > eps) return false;
        }
        return true;
    }

    const Identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

    test("euler → quat → euler roundtrip", () => {
        fc.assert(
            fc.property(safeEulerArb, ({ x, y, z }) => {
                const q = math.quat(x, y, z);
                const e = math.euler(q.x, q.y, q.z, q.w);
                expect(Math.abs(e.x - x)).toBeLessThan(EulerRt);
                expect(Math.abs(e.y - y)).toBeLessThan(EulerRt);
                expect(Math.abs(e.z - z)).toBeLessThan(EulerRt);
            }),
            { numRuns: 500 },
        );
    });

    test("slerp endpoints", () => {
        fc.assert(
            fc.property(unitQuatArb, unitQuatArb, (a, b) => {
                const at0 = math.slerp(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, 0);
                expect(1 - Math.abs(quatDot(at0, a))).toBeLessThan(QuatExact);

                const at1 = math.slerp(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, 1);
                expect(1 - Math.abs(quatDot(at1, b))).toBeLessThan(QuatExact);
            }),
            { numRuns: 500 },
        );
    });

    test("slerp preserves unit norm", () => {
        fc.assert(
            fc.property(unitQuatArb, unitQuatArb, tArb, (a, b, t) => {
                const r = math.slerp(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, t);
                expect(Math.abs(quatNorm(r) - 1)).toBeLessThan(SlerpNorm);
            }),
            { numRuns: 500 },
        );
    });

    test("invert roundtrip", () => {
        fc.assert(
            fc.property(perspectiveMatrixArb, (m) => {
                const inv = math.invert(m);
                const roundtrip = math.invert(inv);
                expect(matClose(roundtrip, m, F32Scaled)).toBe(true);
            }),
            { numRuns: 500 },
        );
    });

    test("multiply associativity", () => {
        fc.assert(
            fc.property(rigidMatrixArb, rigidMatrixArb, rigidMatrixArb, (a, b, c) => {
                const abC = math.multiply(math.multiply(a, b), c);
                const aBc = math.multiply(a, math.multiply(b, c));
                expect(matClose(abC, aBc, F32Scaled)).toBe(true);
            }),
            { numRuns: 500 },
        );
    });

    test("M * M⁻¹ ≈ identity", () => {
        fc.assert(
            fc.property(perspectiveMatrixArb, (m) => {
                const inv = math.invert(m);
                const product = math.multiply(m, inv);
                expect(matClose(product, Identity, F32Scaled)).toBe(true);
            }),
            { numRuns: 500 },
        );
    });

    test("lookAt orthonormality", () => {
        fc.assert(
            fc.property(rigidMatrixArb, (m) => {
                // Column vectors of rotation part
                const cols = [
                    [m[0], m[1], m[2]],
                    [m[4], m[5], m[6]],
                    [m[8], m[9], m[10]],
                ];

                for (const col of cols) {
                    const len = Math.sqrt(col[0] ** 2 + col[1] ** 2 + col[2] ** 2);
                    expect(Math.abs(len - 1)).toBeLessThan(F32Unit);
                }

                for (let i = 0; i < 3; i++) {
                    for (let j = i + 1; j < 3; j++) {
                        const dot =
                            cols[i][0] * cols[j][0] +
                            cols[i][1] * cols[j][1] +
                            cols[i][2] * cols[j][2];
                        expect(Math.abs(dot)).toBeLessThan(F32Unit);
                    }
                }
            }),
            { numRuns: 500 },
        );
    });
});
