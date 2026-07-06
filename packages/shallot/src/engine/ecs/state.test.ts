import { beforeEach, describe, expect, test } from "bun:test";
import { angle, degrees, radians, State } from "../..";
import { entity } from "./component";
import {
    clear,
    dependencies,
    dump,
    find,
    inspect,
    isSingleton,
    readFields,
    register,
    schema,
    snapshot,
} from "./core";
import { schemas } from "./reflection";
import { sparse } from "./sparse";

const formatHex = Object.assign((n: number) => "0x" + (n >>> 0).toString(16).padStart(6, "0"), {
    kind: "color" as const,
});

describe("State", () => {
    let state: State;

    beforeEach(() => {
        state = new State();
    });

    describe("System Registration", () => {
        test("should register a system", () => {
            let called = false;
            state.addSystem({
                group: "simulation",
                update: () => {
                    called = true;
                },
            });

            state.step();
            expect(called).toBe(true);
        });

        test("should unregister a system", () => {
            let count = 0;
            const system = {
                group: "simulation" as const,
                update: () => {
                    count++;
                },
            };

            state.addSystem(system);
            state.step();
            expect(count).toBe(1);

            state.removeSystem(system);
            state.step();
            expect(count).toBe(1);
        });
    });

    describe("Time Tracking", () => {
        test("should have correct initial time values", () => {
            expect(state.time.deltaTime).toBe(0);
            expect(state.time.fixedDeltaTime).toBe(1 / 60);
            expect(state.time.elapsed).toBe(0);
        });

        test("should update elapsed time after step", () => {
            // 1/60 < the clamp cap, so elapsed accumulates the unclamped dt; FP
            // doubling makes the second sum exactly 2/60, so equality is exact
            state.step(1 / 60);
            expect(state.time.elapsed).toBe(1 / 60);

            state.step(1 / 60);
            expect(state.time.elapsed).toBe(2 / 60);
        });
    });

    describe("Dispose", () => {
        test("should call dispose on all systems", () => {
            let disposed = false;
            state.addSystem({
                dispose: () => {
                    disposed = true;
                },
            });

            state.dispose();
            expect(disposed).toBe(true);
        });

        test("double dispose tears systems down only once", () => {
            let disposeCount = 0;
            state.addSystem({ dispose: () => disposeCount++ });

            expect(state.disposed).toBe(false);
            state.dispose();
            state.dispose();

            // the second call short-circuits on the _disposed guard, which `disposed` exposes (the seam an
            // async plugin step checks before touching a torn-down State)
            expect(disposeCount).toBe(1);
            expect(state.disposed).toBe(true);
        });
    });

    describe("Identity", () => {
        test("records the authored set and each entity's scene id", () => {
            const a = state.create();
            const b = state.create();
            state.identity.author(a, "cube");
            state.identity.author(b); // anonymous — authored but unnamed

            expect([...state.identity.authored]).toEqual([a, b]);
            expect(state.identity.id(a)).toBe("cube");
            expect(state.identity.id(b)).toBeUndefined();
        });

        test("destroy forgets identity, so a recycled eid never inherits a stale id", () => {
            const a = state.create();
            state.identity.author(a, "cube");
            state.destroy(a);

            expect(state.identity.id(a)).toBeUndefined();
            expect(state.identity.authored.has(a)).toBe(false);

            // the freelist hands the same slot to the next entity — it must come back clean
            const recycled = state.create();
            expect(recycled).toBe(a);
            expect(state.identity.id(recycled)).toBeUndefined();
        });
    });
});

const Vec3Component = {
    posX: [] as number[],
    posY: [] as number[],
    posZ: [] as number[],
    speed: [] as number[],
};

const ShapeMode = { Square: 0, Circle: 1, Triangle: 2 } as const;

const StyleComponent = {
    color: [] as number[],
    mode: [] as number[],
    tilt: [] as number[],
    target: sparse(entity),
};

describe("Introspection", () => {
    let state: State;

    beforeEach(() => {
        clear();
        register("Vec3Component", Vec3Component, {
            defaults: () => ({ posX: 0, posY: 0, posZ: 0, speed: 1 }),
        });
        register("StyleComponent", StyleComponent, {
            defaults: () => ({ color: 0xff0000, mode: 0, tilt: 1 }),
            format: { color: formatHex },
            parse: { color: (v: string) => Number.parseInt(v.replace("0x", ""), 16) },
            enums: { mode: ShapeMode },
            inputs: { tilt: angle },
        });
        state = new State();
    });

    describe("schema", () => {
        test("returns null for unknown component", () => {
            expect(schema("nonexistent")).toBeNull();
        });

        test("detects scalar fields", () => {
            const s = schema("vec3component")!;
            const speed = s.fields.find((f) => f.name === "speed");
            expect(speed).toBeDefined();
            expect(speed!.kind).toBe("float");
            expect(speed!.default).toBe(1);
        });

        test("detects color fields", () => {
            const s = schema("style-component")!;
            const color = s.fields.find((f) => f.name === "color");
            expect(color).toBeDefined();
            expect(color!.kind).toBe("color");
        });

        test("detects enum fields", () => {
            const s = schema("style-component")!;
            const mode = s.fields.find((f) => f.name === "mode");
            expect(mode).toBeDefined();
            expect(mode!.kind).toBe("enum");
            expect(mode!.options).toEqual(ShapeMode);
        });

        test("detects entity refs from the field's type — never a bare float", () => {
            const s = schema("style-component")!;
            const target = s.fields.find((f) => f.name === "target");
            expect(target).toBeDefined();
            expect(target!.kind).toBe("entity");
        });

        test("detects unit fields from the angle input and carries the unit menu", () => {
            const s = schema("style-component")!;
            const tilt = s.fields.find((f) => f.name === "tilt");
            expect(tilt).toBeDefined();
            expect(tilt!.kind).toBe("unit");
            expect(tilt!.units).toEqual([degrees, radians]);
        });

        test("schemas returns all registered", () => {
            const all = schemas();
            const names = all.map((s) => s.name);
            expect(names.sort()).toEqual(["style-component", "vec3component"]);
        });
    });

    describe("readFields", () => {
        test("reads numeric fields from component", () => {
            const eid = state.create();
            state.add(eid, Vec3Component);
            Vec3Component.posX[eid] = 10;
            Vec3Component.posY[eid] = 20;
            Vec3Component.posZ[eid] = 30;
            Vec3Component.speed[eid] = 5;

            const fields = readFields(Vec3Component, eid);
            expect(fields.posX).toBe(10);
            expect(fields.posY).toBe(20);
            expect(fields.posZ).toBe(30);
            expect(fields.speed).toBe(5);
        });
    });

    describe("inspect", () => {
        test("returns null for nonexistent entity", () => {
            expect(inspect(state, 999)).toBeNull();
        });

        test("returns all components for an entity", () => {
            const eid = state.create();
            state.add(eid, Vec3Component);
            state.add(eid, StyleComponent);
            Vec3Component.posX[eid] = 7;
            StyleComponent.color[eid] = 0x00ff00;

            const data = inspect(state, eid)!;
            expect(data).not.toBeNull();
            expect(data.eid).toBe(eid);
            expect(data.components["vec3component"]).toBeDefined();
            expect(data.components["vec3component"].posX).toBe(7);
            expect(data.components["style-component"]).toBeDefined();
            expect(data.components["style-component"].color).toBe(0x00ff00);
        });
    });

    describe("find", () => {
        test("returns empty for unknown component", () => {
            expect(find(state, "nonexistent")).toEqual([]);
        });

        test("finds entities with a component", () => {
            const e1 = state.create();
            const e2 = state.create();
            const e3 = state.create();
            state.add(e1, Vec3Component);
            state.add(e2, Vec3Component);
            state.add(e3, StyleComponent);

            const results = find(state, "vec3component");
            expect(results.length).toBe(2);
            expect(results.map((r) => r.eid).sort()).toEqual([e1, e2].sort());
        });
    });

    describe("snapshot", () => {
        test("captures full world state", () => {
            const e1 = state.create();
            const e2 = state.create();
            state.add(e1, Vec3Component);
            state.add(e2, StyleComponent);
            Vec3Component.speed[e1] = 42;

            const snap = snapshot(state);
            expect(snap.length).toBe(2);

            const d1 = snap.find((d) => d.eid === e1)!;
            expect(d1.components["vec3component"].speed).toBe(42);

            const d2 = snap.find((d) => d.eid === e2)!;
            expect(d2.components["style-component"]).toBeDefined();
        });
    });

    describe("dump", () => {
        test("returns not found for missing entity", () => {
            expect(dump(state, 999)).toBe("Entity 999: not found");
        });

        test("returns human-readable string", () => {
            const eid = state.create();
            state.add(eid, Vec3Component);
            Vec3Component.posX[eid] = 1;
            Vec3Component.speed[eid] = 5;

            const result = dump(state, eid);
            expect(result).toContain(`Entity ${eid}:`);
            expect(result).toContain("vec3component:");
            expect(result).toContain("speed: 5");
        });
    });
});

describe("dependencies()", () => {
    const A = { value: [] as number[] };
    const B = { value: [] as number[] };
    const C = { value: [] as number[] };

    beforeEach(() => {
        clear();
    });

    test("returns required component names", () => {
        register("B", B);
        register("C", C);
        register("A", A, { requires: [B, C] });

        expect(dependencies("a")).toEqual(["b", "c"]);
    });

    test("returns empty for no requires", () => {
        register("A", A);
        expect(dependencies("a")).toEqual([]);
    });

    test("returns empty for unknown component", () => {
        expect(dependencies("nonexistent")).toEqual([]);
    });
});

describe("isSingleton()", () => {
    const A = { value: [] as number[] };
    const B = { value: [] as number[] };

    beforeEach(() => {
        clear();
    });

    test("reads the trait", () => {
        register("A", A, { singleton: true });
        expect(isSingleton("a")).toBe(true);
    });

    test("defaults false without the trait", () => {
        register("B", B);
        expect(isSingleton("b")).toBe(false);
    });

    test("false for unknown component", () => {
        expect(isSingleton("nonexistent")).toBe(false);
    });
});
