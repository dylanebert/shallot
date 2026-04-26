import { describe, test, expect, beforeEach } from "bun:test";
import {
    build,
    run,
    type State,
    Transform,
    WorldTransform,
    TransformsPlugin,
    InputPlugin,
    Part,
    Tween,
    TweenState,
    TweenPlugin,
    parse,
    load,
    type Plugin,
    type System,
} from "../src";
import { createTween } from "../src/extras/tween/tween";
import { clearRegistry, getComponent } from "../src/engine/ecs/component";
import { Orbit, OrbitPlugin } from "../src/extras/orbit";
import { clearRelations } from "../src/engine/ecs/relation";
import { count, first, all, stepFor } from "./helpers/state";

const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };

const TestPlugin: Plugin = {
    name: "Test",
    components: { Position },
};

function initTransform(eid: number): void {
    Transform.posX[eid] = 0;
    Transform.posY[eid] = 0;
    Transform.posZ[eid] = 0;
    Transform.rotX[eid] = 0;
    Transform.rotY[eid] = 0;
    Transform.rotZ[eid] = 0;
    Transform.scaleX[eid] = 1;
    Transform.scaleY[eid] = 1;
    Transform.scaleZ[eid] = 1;
}

function initOrbit(
    eid: number,
    overrides: Partial<{ distance: number; pitch: number; yaw: number }> = {},
): void {
    const distance = overrides.distance ?? 10;
    const pitch = overrides.pitch ?? 0;
    const yaw = overrides.yaw ?? 0;
    Orbit.distance[eid] = distance;
    Orbit.pitch[eid] = pitch;
    Orbit.yaw[eid] = yaw;
    Orbit.minPitch[eid] = -Math.PI / 2 + 0.01;
    Orbit.maxPitch[eid] = Math.PI / 2 - 0.01;
    Orbit.minDistance[eid] = 1;
    Orbit.maxDistance[eid] = 100;
    Orbit.smoothness[eid] = 0.15;
    Orbit.sensitivity[eid] = 0.005;
    Orbit.zoomSpeed[eid] = 0.001;
    Orbit.orbitButton[eid] = 0;
}

describe("Lifecycle", () => {
    beforeEach(() => {
        clearRegistry();
        clearRelations();
    });

    describe("component added mid-session", () => {
        test("Part — query includes entity after mid-session add", async () => {
            const state = await build({
                plugins: [TransformsPlugin],
                defaults: false,
            });

            const eid = state.addEntity();
            state.addComponent(eid, Transform);
            initTransform(eid);
            state.step();
            state.step();

            state.addComponent(eid, Part);

            state.step();

            expect(count(state, [Part, WorldTransform])).toBeGreaterThanOrEqual(1);
            const eids = all(state, [Part, WorldTransform]);
            expect(eids).toContain(eid);
        });

        test("Orbit — system updates Transform on next step", async () => {
            const state = await build({
                plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
                defaults: false,
            });

            const eid = state.addEntity();
            state.addComponent(eid, Transform);
            initTransform(eid);
            state.step();
            state.step();

            state.addComponent(eid, Orbit);
            initOrbit(eid, { distance: 10, pitch: 0, yaw: 0 });

            state.step();

            expect(Transform.posZ[eid]).toBeCloseTo(10, 1);
        });

        test("Tween — field accessor resolves, interpolation starts", async () => {
            const state = await build({
                plugins: [TweenPlugin, TestPlugin],
                defaults: false,
            });

            const eid = state.addEntity();
            state.addComponent(eid, Position);
            Position.x[eid] = 0;
            state.step();
            state.step();

            const tweenEid = createTween(state, eid, "position.x", {
                to: 100,
                duration: 1,
            });
            expect(tweenEid).not.toBeNull();
            Tween.state[tweenEid!] = TweenState.Playing;

            stepFor(state, 0.5);

            expect(Position.x[eid]).toBeCloseTo(50, 1);
        });
    });

    describe("entity destroyed during iteration", () => {
        test("despawned entity absent from all queries after step", async () => {
            const state = await build({
                plugins: [TransformsPlugin, TestPlugin],
                defaults: false,
            });

            const e1 = state.addEntity();
            const e2 = state.addEntity();
            const e3 = state.addEntity();
            for (const eid of [e1, e2, e3]) {
                state.addComponent(eid, Transform);
                state.addComponent(eid, Position);
            }

            state.step();
            expect(count(state, [Transform])).toBe(3);

            state.removeEntity(e2);
            state.step();

            const remaining = all(state, [Transform]);
            expect(remaining).toContain(e1);
            expect(remaining).not.toContain(e2);
            expect(remaining).toContain(e3);
            expect(count(state, [Position])).toBe(2);
        });

        test("system survives despawn between steps", async () => {
            const processed: number[] = [];

            const TrackingSystem: System = {
                update(state: State) {
                    processed.length = 0;
                    for (const eid of state.query([Position])) {
                        Position.x[eid] += 1;
                        processed.push(eid);
                    }
                },
            };

            const TrackingPlugin: Plugin = {
                name: "Tracking",
                components: { Position },
                systems: [TrackingSystem],
            };

            const state = await build({
                plugins: [TrackingPlugin],
                defaults: false,
            });

            const eids: number[] = [];
            for (let i = 0; i < 5; i++) {
                const eid = state.addEntity();
                state.addComponent(eid, Position);
                Position.x[eid] = 0;
                eids.push(eid);
            }

            state.step();
            expect(processed.length).toBe(5);
            expect(Position.x[eids[0]]).toBe(1);

            state.removeEntity(eids[2]);
            state.step();

            expect(processed.length).toBe(4);
            expect(processed).not.toContain(eids[2]);
            expect(Position.x[eids[0]]).toBe(2);
            expect(Position.x[eids[4]]).toBe(2);
        });

        test("tween targeting destroyed entity does not crash", async () => {
            const state = await build({
                plugins: [TweenPlugin, TestPlugin],
                defaults: false,
            });

            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });
            Tween.state[tweenEid!] = TweenState.Playing;

            stepFor(state, 0.25);
            expect(Position.x[target]).toBeCloseTo(25, 1);

            state.removeEntity(target);

            expect(() => stepFor(state, 0.25)).not.toThrow();
        });
    });

    describe("plugin toggle lifecycle", () => {
        test("build without plugin — component not registered, load skips attr", async () => {
            const Health = { current: [] as number[], max: [] as number[] };
            const HealthPlugin: Plugin = { name: "Health", components: { Health } };

            const nodes = parse(
                `<scene>
                    <a transform="pos: 1 2 3" health="current: 50; max: 100" />
                </scene>`,
            );

            const state1 = await build({
                plugins: [TransformsPlugin, HealthPlugin],
                defaults: false,
            });
            load(nodes, state1);

            expect(getComponent("health")).toBeDefined();
            expect(count(state1, [Health])).toBe(1);
            const eid1 = first(state1, [Health]);
            expect(Health.current[eid1]).toBe(50);
            expect(Health.max[eid1]).toBe(100);

            state1.dispose();
            clearRegistry();

            const state2 = await build({
                plugins: [TransformsPlugin],
                defaults: false,
            });
            load(nodes, state2);

            expect(getComponent("health")).toBeUndefined();
            expect(count(state2, [Transform])).toBe(1);
        });

        test("re-register plugin, reload same nodes — component data recovered", async () => {
            const Health = { current: [] as number[], max: [] as number[] };
            const HealthPlugin: Plugin = { name: "Health", components: { Health } };

            const nodes = parse(
                `<scene>
                    <a transform="pos: 1 2 3" health="current: 50; max: 100" />
                </scene>`,
            );

            const state1 = await build({
                plugins: [TransformsPlugin, HealthPlugin],
                defaults: false,
            });
            load(nodes, state1);
            expect(count(state1, [Health])).toBe(1);

            state1.dispose();
            clearRegistry();

            const state2 = await build({
                plugins: [TransformsPlugin],
                defaults: false,
            });
            load(nodes, state2);
            expect(getComponent("health")).toBeUndefined();

            state2.dispose();
            clearRegistry();

            const state3 = await build({
                plugins: [TransformsPlugin, HealthPlugin],
                defaults: false,
            });
            load(nodes, state3);

            expect(getComponent("health")).toBeDefined();
            expect(count(state3, [Health])).toBe(1);
            const eid = first(state3, [Health]);
            expect(Health.current[eid]).toBe(50);
            expect(Health.max[eid]).toBe(100);
        });
    });

    describe("dispose correctness", () => {
        test("dispose fires onDispose hooks and system.dispose()", async () => {
            let hookCalled = false;
            let systemDisposed = false;

            const DisposableSystem: System = {
                update() {},
                dispose() {
                    systemDisposed = true;
                },
            };

            const state = await build({
                plugins: [{ name: "Disposable", systems: [DisposableSystem] }],
                defaults: false,
            });

            state.onDispose(() => {
                hookCalled = true;
            });

            state.step();
            state.step();
            state.dispose();

            expect(hookCalled).toBe(true);
            expect(systemDisposed).toBe(true);
        });

        test("stepping after dispose does not crash", async () => {
            let runCount = 0;

            const CountingSystem: System = {
                update() {
                    runCount++;
                },
            };

            const state = await build({
                plugins: [{ name: "Counting", systems: [CountingSystem] }],
                defaults: false,
            });

            state.step();
            expect(runCount).toBe(1);

            state.dispose();

            expect(() => state.step()).not.toThrow();
        });

        test("build after dispose produces clean state", async () => {
            const state1 = await build({
                plugins: [TransformsPlugin],
                defaults: false,
            });

            const eid = state1.addEntity();
            state1.addComponent(eid, Transform);
            state1.step();

            expect(count(state1, [Transform])).toBe(1);
            state1.dispose();
            clearRegistry();

            const state2 = await build({
                plugins: [TransformsPlugin],
                defaults: false,
            });

            expect(state2.getAllEntities().length).toBe(0);
            expect(count(state2, [Transform])).toBe(0);
        });

        test("dispose stops the frame loop", async () => {
            const state = await run({
                plugins: [TransformsPlugin],
                defaults: false,
            });

            await new Promise((r) => setTimeout(r, 20));
            state.dispose();
        });
    });
});
