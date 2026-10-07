import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { BodyFlags } from "../world/body";
import { BodyField, bodyField } from "./bodyrecords";
import { S2_CENTER0, S2_FLAGS, S2_MIN_EXTENT, SIM_STRIDE, STATE_STRIDE } from "./columns";
import gold from "./finalize.gold.json";
import { kernel } from "./kernel";

const S2_ROTATION0 = 10;
const simMap = [
    26,
    51,
    49,
    50,
    20,
    21,
    22,
    23,
    24,
    25,
    ...Array.from({ length: 9 }, (_, i) => 27 + i),
    ...Array.from({ length: 9 }, (_, i) => 36 + i),
    3,
    4,
    5,
    6,
];
const finMap = [7, 8, 9, 17, 18, 19, 46, 47, 48, 0, 1, 2];
const bits = new Uint32Array(1);
const float = new Float32Array(bits.buffer);
function decode(hex: string): number {
    bits[0] = Number.parseInt(hex.slice(2), 16);
    return float[0];
}
function assertBits(value: number, hex: string) {
    float[0] = value;
    expect(bits[0]).toBe(Number.parseInt(hex.slice(2), 16));
}

function fixture(count: number, dt: number, continuous: boolean, invDt = 1 / dt) {
    const world = new PhysicsWorld({
        gravity: { x: 0, y: 0, z: 0 },
        enableSleep: false,
        enableContinuous: continuous,
    });
    for (let i = 0; i < count; ++i) world.createBody({ type: BodyType.Dynamic });
    // Initializes the real step context, arenas and finalize bookkeeping before the direct task run.
    world.step(dt, 1);
    const k = kernel(world.state.ecsState);
    // The frozen phase inputs supply dt and invDt independently.
    const context = new Float32Array(k.memory.buffer, k.stepContext(dt, 1, 0, 0), 10);
    context[1] = invDt;
    k.stepSolveBuild(1, 1, 0, 0, 0, 1000, 0, false, 0, 0, continuous, false);
    world.state.bodyStore.refreshViews();
    return { world, k, store: world.state.bodyStore };
}
function finalize(k: ReturnType<typeof kernel>, count: number) {
    // The existing WASM finalize job (7) is outside the coordinator's narrower ParKind union.
    (k.parBuild as (kind: number, count: number, threads: number, a: number) => number)(
        7,
        count,
        1,
        0,
    );
    k.runMt();
}

function load(store: ReturnType<typeof fixture>["store"], c: (typeof gold.cases)[number]) {
    for (let i = 0; i < simMap.length; ++i) store.simF[simMap[i]] = decode(c.sim[i]);
    for (let i = 0; i < finMap.length; ++i) store.simF[finMap[i]] = decode(c.fin[i]);
    store.stateF.set(c.state.map(decode));
}

test("finalize maxMotion gold decides the exact fast boundary", () => {
    for (const c of gold.cases) {
        for (const below of [false, true]) {
            const { world, k, store } = fixture(1, decode(c.h), true, decode(c.invDt));
            try {
                load(store, c);
                float[0] = 2 * decode(c.out[1]);
                if (below) --bits[0];
                store.simF[S2_MIN_EXTENT] = float[0];
                store.flagsU[13] = BodyFlags.dynamicFlag;
                store.simF.set(store.simF.slice(3, 7), S2_ROTATION0);
                store.simF.set(store.simF.slice(7, 10), S2_CENTER0);
                finalize(k, 1);
                expect((store.sim2U[S2_FLAGS] & BodyFlags.isFast) !== 0).toBe(below);
                assertBits(bodyField(world.state, 0, BodyField.sleepVelocity), c.out[0]);
            } finally {
                world.destroy();
            }
        }
    }
});

test("finalize matches frozen C vectors through resident WASM records", () => {
    expect(gold.cases.length).toBeGreaterThan(0);
    for (const c of gold.cases) {
        const { world, k, store } = fixture(1, decode(c.h), true, decode(c.invDt));
        try {
            load(store, c);
            store.flagsU[13] = 0;
            finalize(k, 1);
            for (let i = 0; i < simMap.length; ++i) assertBits(store.simF[simMap[i]], c.outSim[i]);
            for (let i = 0; i < finMap.length; ++i) assertBits(store.simF[finMap[i]], c.outFin[i]);
            for (let i = 0; i < c.outState.length; ++i) assertBits(store.stateF[i], c.outState[i]);
            assertBits(bodyField(world.state, 0, BodyField.sleepVelocity), c.out[0]);
            for (let i = 0; i < 3; ++i) assertBits(store.simF[S2_CENTER0 + i], c.outFin[i]);
            for (let i = 0; i < 4; ++i) assertBits(store.simF[S2_ROTATION0 + i], c.outSim[28 + i]);
        } finally {
            world.destroy();
        }
    }
});

for (const continuous of [true, false]) {
    for (const [index, minExtent] of [0.5, 2].entries()) {
        test(`fast-candidate predicate: continuous=${continuous}, minExtent=${minExtent}`, () => {
            const { world, k, store } = fixture(2, 1, continuous);
            try {
                for (let i = 0; i < 2; ++i) {
                    const s = i * SIM_STRIDE,
                        t = i * STATE_STRIDE;
                    store.simF.fill(0, s, s + 52);
                    store.simF[s + 6] = 1;
                    store.simF.set([1, 2, 3], s + 7);
                    store.simF.fill(99, s + S2_ROTATION0, s + S2_CENTER0 + 3);
                    store.simF[s + S2_MIN_EXTENT] = i === 0 ? 0.5 : 2;
                    store.stateF.fill(0, t, t + 13);
                    store.stateF[t + 6] = 0.4;
                    store.stateF[t + 12] = 1;
                }
                finalize(k, 2);
                const s = index * SIM_STRIDE;
                expect((store.sim2U[s + S2_FLAGS] & BodyFlags.isFast) !== 0).toBe(
                    continuous && index === 0,
                );
                // Immediate continuous solve and the non-fast branch both commit the resulting base.
                expect(Array.from(store.simF.slice(s + S2_CENTER0, s + S2_CENTER0 + 3))).toEqual(
                    Array.from(store.simF.slice(s + 7, s + 10)),
                );
                expect(
                    Array.from(store.simF.slice(s + S2_ROTATION0, s + S2_ROTATION0 + 4)),
                ).toEqual(Array.from(store.simF.slice(s + 3, s + 7)));
                expect(bodyField(world.state, index, BodyField.sleepVelocity)).toBe(
                    Math.fround(0.2),
                );
            } finally {
                world.destroy();
            }
        });
    }
}
