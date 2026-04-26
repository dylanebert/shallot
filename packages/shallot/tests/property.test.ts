import { test, expect, describe, beforeEach } from "bun:test";
import fc from "fast-check";
import {
    State,
    TransformsPlugin,
    RenderPlugin,
    Transform,
    Part,
    Camera,
    parseFields,
    formatFields,
    type Plugin,
    type Component,
} from "../src";
import { schemas } from "../src/engine/ecs/reflection";
import { clearRegistry } from "../src/engine/ecs/component";
import { clearRelations } from "../src/engine/ecs/relation";
import * as math from "../src/engine/utils/math";
import { all } from "./helpers/state";

const PLUGINS: Plugin[] = [TransformsPlugin, RenderPlugin];

const COMPONENTS: { component: Component; name: string }[] = [
    { component: Transform, name: "Transform" },
    { component: Part, name: "Part" },
    { component: Camera, name: "Camera" },
];

function setup() {
    clearRegistry();
    clearRelations();
    const state = new State();
    for (const plugin of PLUGINS) state.register(plugin);
    return state;
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
                const state = setup();
                const alive: number[] = [];
                const despawned = new Set<number>();
                const entityComponents = new Map<number, Set<Component>>();

                for (const cmd of commands) {
                    switch (cmd.type) {
                        case "spawn": {
                            const eid = state.addEntity();
                            alive.push(eid);
                            despawned.delete(eid);
                            entityComponents.set(eid, new Set());
                            break;
                        }
                        case "despawn": {
                            if (alive.length === 0) break;
                            const idx = cmd.entityIdx % alive.length;
                            const eid = alive[idx];
                            state.removeEntity(eid);
                            alive.splice(idx, 1);
                            despawned.add(eid);
                            entityComponents.delete(eid);
                            break;
                        }
                        case "addComponent": {
                            if (alive.length === 0) break;
                            const eid = alive[cmd.entityIdx % alive.length];
                            const { component } = COMPONENTS[cmd.componentIdx];
                            state.addComponent(eid, component as never);
                            entityComponents.get(eid)!.add(component);
                            break;
                        }
                        case "removeComponent": {
                            if (alive.length === 0) break;
                            const eid = alive[cmd.entityIdx % alive.length];
                            const { component } = COMPONENTS[cmd.componentIdx];
                            if (state.hasComponent(eid, component as never)) {
                                state.removeComponent(eid, component as never);
                                entityComponents.get(eid)!.delete(component);
                            }
                            break;
                        }
                    }

                    // Invariant 2: despawned entities absent
                    for (const eid of despawned) {
                        expect(state.entityExists(eid)).toBe(false);
                    }

                    // Invariant 3: query consistency
                    for (const { component } of COMPONENTS) {
                        const queryResult = new Set(all(state, [component]));
                        for (const eid of alive) {
                            const has = state.hasComponent(eid, component as never);
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
    beforeEach(() => {
        setup();
    });

    test("parseFields(formatFields(fields)) preserves values", () => {
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
                            if (a === 0) {
                                expect(Math.abs(b)).toBeLessThan(1e-5);
                            } else {
                                expect(Math.abs(b - a) / Math.max(1, Math.abs(a))).toBeLessThan(
                                    1e-5,
                                );
                            }
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
    const Epsilon = 1e-3;

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
        .map(([ex, ey, ez, tx, ty, tz]) => math.lookAtMatrix(ex, ey, ez, tx, ty, tz));

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
                const q = math.eulerToQuaternion(x, y, z);
                const e = math.quaternionToEuler(q.x, q.y, q.z, q.w);
                expect(Math.abs(e.x - x)).toBeLessThan(Epsilon);
                expect(Math.abs(e.y - y)).toBeLessThan(Epsilon);
                expect(Math.abs(e.z - z)).toBeLessThan(Epsilon);
            }),
            { numRuns: 500 },
        );
    });

    test("rotate preserves unit norm", () => {
        fc.assert(
            fc.property(unitQuatArb, safeEulerArb, (q, { x, y, z }) => {
                const r = math.rotate(q.x, q.y, q.z, q.w, x, y, z);
                expect(Math.abs(quatNorm(r) - 1)).toBeLessThan(Epsilon);
            }),
            { numRuns: 500 },
        );
    });

    test("slerp endpoints", () => {
        fc.assert(
            fc.property(unitQuatArb, unitQuatArb, (a, b) => {
                const at0 = math.slerp(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, 0);
                expect(Math.abs(quatDot(at0, a)) - 1).toBeLessThan(Epsilon);

                const at1 = math.slerp(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, 1);
                expect(Math.abs(quatDot(at1, b)) - 1).toBeLessThan(Epsilon);
            }),
            { numRuns: 500 },
        );
    });

    test("slerp preserves unit norm", () => {
        fc.assert(
            fc.property(unitQuatArb, unitQuatArb, tArb, (a, b, t) => {
                const r = math.slerp(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, t);
                expect(Math.abs(quatNorm(r) - 1)).toBeLessThan(Epsilon);
            }),
            { numRuns: 500 },
        );
    });

    test("invertMatrix roundtrip", () => {
        fc.assert(
            fc.property(perspectiveMatrixArb, (m) => {
                const inv = math.invertMatrix(m);
                const roundtrip = math.invertMatrix(inv);
                expect(matClose(roundtrip, m, 1e-3)).toBe(true);
            }),
            { numRuns: 500 },
        );
    });

    test("rigid inverse roundtrip", () => {
        fc.assert(
            fc.property(rigidMatrixArb, (m) => {
                const roundtrip = math.invert(math.invert(m));
                expect(matClose(roundtrip, m, 1e-3)).toBe(true);
            }),
            { numRuns: 500 },
        );
    });

    test("multiply associativity", () => {
        fc.assert(
            fc.property(rigidMatrixArb, rigidMatrixArb, rigidMatrixArb, (a, b, c) => {
                const abC = math.multiply(math.multiply(a, b), c);
                const aBc = math.multiply(a, math.multiply(b, c));
                expect(matClose(abC, aBc, 1e-3)).toBe(true);
            }),
            { numRuns: 500 },
        );
    });

    test("M * M⁻¹ ≈ identity", () => {
        fc.assert(
            fc.property(perspectiveMatrixArb, (m) => {
                const inv = math.invertMatrix(m);
                const product = math.multiply(m, inv);
                expect(matClose(product, Identity, 1e-3)).toBe(true);
            }),
            { numRuns: 500 },
        );
    });

    test("lookAtMatrix orthonormality", () => {
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
                    expect(Math.abs(len - 1)).toBeLessThan(Epsilon);
                }

                for (let i = 0; i < 3; i++) {
                    for (let j = i + 1; j < 3; j++) {
                        const dot =
                            cols[i][0] * cols[j][0] +
                            cols[i][1] * cols[j][1] +
                            cols[i][2] * cols[j][2];
                        expect(Math.abs(dot)).toBeLessThan(Epsilon);
                    }
                }
            }),
            { numRuns: 500 },
        );
    });
});
