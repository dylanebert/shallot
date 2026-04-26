import { test, expect, describe, beforeEach } from "bun:test";
import { State, traits } from "../src";
import { formatHex } from "../src/engine/ecs/strings";
import {
    schema,
    schemas,
    dependencies,
    readFields,
    inspect,
    find,
    snapshot,
    dump,
} from "../src/engine/ecs/reflection";
import { clearRegistry, registerComponent } from "../src/engine/ecs/component";

const TestComponent = { value: [] as number[] };

describe("State", () => {
    let state: State;

    beforeEach(() => {
        state = new State();
    });

    describe("Entity Lifecycle", () => {
        test("should create an entity", () => {
            const entity = state.addEntity();
            expect(entity).toBeGreaterThanOrEqual(0);
            expect(state.entityExists(entity)).toBe(true);
        });

        test("should destroy an entity", () => {
            const entity = state.addEntity();
            expect(state.entityExists(entity)).toBe(true);

            state.removeEntity(entity);
            expect(state.entityExists(entity)).toBe(false);
        });

        test("should create multiple entities", () => {
            const e1 = state.addEntity();
            const e2 = state.addEntity();
            const e3 = state.addEntity();

            expect(e1).not.toBe(e2);
            expect(e2).not.toBe(e3);
            expect(state.entityExists(e1)).toBe(true);
            expect(state.entityExists(e2)).toBe(true);
            expect(state.entityExists(e3)).toBe(true);
        });

        test("should grow capacity beyond initial limit", () => {
            for (let i = 0; i < 2000; i++) {
                state.addEntity();
            }
            expect(state.max).toBeGreaterThanOrEqual(2000);
        });

        test("should track max entity ID", () => {
            expect(state.max).toBe(0);

            const e1 = state.addEntity();
            expect(state.max).toBe(e1);

            const e2 = state.addEntity();
            expect(state.max).toBe(e2);

            state.removeEntity(e2);
            expect(state.max).toBe(e1);

            const e3 = state.addEntity();
            expect(state.max).toBe(e3);
        });

        test("should shrink max after bulk removal", () => {
            const eids: number[] = [];
            for (let i = 0; i < 100; i++) eids.push(state.addEntity());
            expect(state.max).toBe(eids[99]);

            for (let i = 99; i >= 10; i--) state.removeEntity(eids[i]);
            expect(state.max).toBe(eids[9]);
        });
    });

    describe("Component Operations", () => {
        test("should add a component to entity", () => {
            const entity = state.addEntity();
            expect(state.hasComponent(entity, TestComponent)).toBe(false);

            state.addComponent(entity, TestComponent);
            expect(state.hasComponent(entity, TestComponent)).toBe(true);
        });

        test("should remove a component from entity", () => {
            const entity = state.addEntity();
            state.addComponent(entity, TestComponent);
            expect(state.hasComponent(entity, TestComponent)).toBe(true);

            state.removeComponent(entity, TestComponent);
            expect(state.hasComponent(entity, TestComponent)).toBe(false);
        });

        test("should set and read component values", () => {
            const entity = state.addEntity();
            state.addComponent(entity, TestComponent);

            TestComponent.value[entity] = 42;
            expect(TestComponent.value[entity]).toBe(42);
        });
    });

    describe("Query Integration", () => {
        test("should query entities with components", () => {
            const entity1 = state.addEntity();
            const entity2 = state.addEntity();
            state.addComponent(entity1, TestComponent);

            const results = [...state.query([TestComponent])];

            expect(results).toContain(entity1);
            expect(results).not.toContain(entity2);
        });

        test("should update query results after component changes", () => {
            const entity = state.addEntity();

            expect([...state.query([TestComponent])]).not.toContain(entity);

            state.addComponent(entity, TestComponent);
            expect([...state.query([TestComponent])]).toContain(entity);

            state.removeComponent(entity, TestComponent);
            expect([...state.query([TestComponent])]).not.toContain(entity);
        });
    });

    describe("System Registration", () => {
        test("should register a system", () => {
            let called = false;
            state.register({
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

            state.register(system);
            state.step();
            expect(count).toBe(1);

            state.unregister(system);
            state.step();
            expect(count).toBe(1);
        });

        test("should expose systems as ReadonlySet", () => {
            const system1 = { update: () => {} };
            const system2 = { update: () => {} };

            state.register(system1);
            state.register(system2);

            expect(state.scheduler.systems.size).toBe(2);
            expect(state.scheduler.systems.has(system1)).toBe(true);
            expect(state.scheduler.systems.has(system2)).toBe(true);
        });
    });

    describe("Time Tracking", () => {
        test("should have correct initial time values", () => {
            expect(state.time.deltaTime).toBe(0);
            expect(state.time.fixedDeltaTime).toBe(1 / 60);
            expect(state.time.elapsed).toBe(0);
        });

        test("should update elapsed time after step", () => {
            state.step(1 / 60);
            expect(state.time.elapsed).toBeCloseTo(1 / 60, 5);

            state.step(1 / 60);
            expect(state.time.elapsed).toBeCloseTo(2 / 60, 5);
        });
    });

    describe("Dispose", () => {
        test("should call dispose on all systems", () => {
            let disposed = false;
            state.register({
                dispose: () => {
                    disposed = true;
                },
            });

            state.dispose();
            expect(disposed).toBe(true);
        });

        test("should be idempotent on double dispose", () => {
            state.dispose();
            expect(() => state.dispose()).not.toThrow();
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
};

traits(StyleComponent, {
    defaults: () => ({ color: 0xff0000, mode: 0 }),
    format: { color: formatHex },
    parse: { color: (v: string) => Number.parseInt(v.replace("0x", ""), 16) },
    enums: { mode: ShapeMode },
});

traits(Vec3Component, {
    defaults: () => ({ posX: 0, posY: 0, posZ: 0, speed: 1 }),
});

describe("Introspection", () => {
    let state: State;

    beforeEach(() => {
        clearRegistry();
        registerComponent("Vec3Component", Vec3Component);
        registerComponent("StyleComponent", StyleComponent);
        state = new State();
    });

    describe("schema", () => {
        test("returns null for unknown component", () => {
            expect(schema("nonexistent")).toBeNull();
        });

        test("detects vec3 fields", () => {
            const s = schema("vec3component")!;
            expect(s).not.toBeNull();
            expect(s.name).toBe("vec3component");

            const pos = s.fields.find((f) => f.name === "pos");
            expect(pos).toBeDefined();
            expect(pos!.kind).toBe("vec3");
            expect(pos!.fields).toEqual(["posX", "posY", "posZ"]);
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

        test("schemas returns all registered", () => {
            const all = schemas();
            expect(all.length).toBe(2);
            expect(all.map((s) => s.name).sort()).toEqual(["style-component", "vec3component"]);
        });
    });

    describe("readFields", () => {
        test("reads numeric fields from component", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Vec3Component);
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
            const eid = state.addEntity();
            state.addComponent(eid, Vec3Component);
            state.addComponent(eid, StyleComponent);
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
            const e1 = state.addEntity();
            const e2 = state.addEntity();
            const e3 = state.addEntity();
            state.addComponent(e1, Vec3Component);
            state.addComponent(e2, Vec3Component);
            state.addComponent(e3, StyleComponent);

            const results = find(state, "vec3component");
            expect(results.length).toBe(2);
            expect(results.map((r) => r.eid).sort()).toEqual([e1, e2].sort());
        });
    });

    describe("snapshot", () => {
        test("captures full world state", () => {
            const e1 = state.addEntity();
            const e2 = state.addEntity();
            state.addComponent(e1, Vec3Component);
            state.addComponent(e2, StyleComponent);
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
            const eid = state.addEntity();
            state.addComponent(eid, Vec3Component);
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
        clearRegistry();
    });

    test("returns required component names", () => {
        traits(A, { requires: [B, C] });
        registerComponent("A", A);
        registerComponent("B", B);
        registerComponent("C", C);

        expect(dependencies("a")).toEqual(["b", "c"]);
    });

    test("returns empty for no requires", () => {
        registerComponent("A", A);
        expect(dependencies("a")).toEqual([]);
    });

    test("returns empty for unknown component", () => {
        expect(dependencies("nonexistent")).toEqual([]);
    });
});
