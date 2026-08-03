import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import * as d from "typegpu/data";
import { flat, integerDiscipline, noDivision } from "../../../../packages/shallot/tests/wgsl";
import { FountainPlugin, FountainSystem } from "./fountain";
import { fountainIntegrateWgsl, hashU32, integrateParticle } from "./kernel";
import {
    cleanupFountainGeneration,
    createFountainLifecycle,
    withOwnedFountainBuffers,
} from "./lifecycle";

const f = Math.fround;
const GRAVITY = f(9.8);
const SPAWN_Y = f(0.05);
const GOLDEN = 0x9e3779b9;

function hashRef(x: number): number {
    let h = x >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    h = Math.imul(h, 0x7feb352d) >>> 0;
    h = (h ^ (h >>> 15)) >>> 0;
    h = Math.imul(h, 0x846ca68b) >>> 0;
    return (h ^ (h >>> 16)) >>> 0;
}

function rndRef(seed: number): number {
    return f(f(hashRef(seed)) * f(1 / 4294967296));
}

function jetRef(seed: number): [number, number, number] {
    const angle = f(rndRef(seed) * f(6.2831853));
    const radius = f(f(Math.sqrt(rndRef((seed + GOLDEN) >>> 0))) * f(1.5));
    const t = rndRef((seed + 0x85ebca6b) >>> 0);
    const up = f(f(f(6.5) * f(1 - t)) + f(f(8) * t));
    return [f(f(Math.cos(angle)) * radius), up, f(f(Math.sin(angle)) * radius)];
}

type Vec4 = [number, number, number, number];

function integrateRef(i: number, posSeed: Vec4, velocity: Vec4, dt: number, frame: number) {
    let pos: [number, number, number] = [f(posSeed[0]), f(posSeed[1]), f(posSeed[2])];
    let vel: [number, number, number] = [f(velocity[0]), f(velocity[1]), f(velocity[2])];
    if (posSeed[3] < f(0.5)) {
        const v0 = jetRef((i + 0x01234567) >>> 0);
        const flight = f(f(f(2) * v0[1]) / GRAVITY);
        const t = f(rndRef((Math.imul(i, 2) + 1) >>> 0) * flight);
        const halfGravityT2 = f(f(f(f(0.5) * f(-GRAVITY)) * t) * t);
        pos = [
            f(f(0 + f(v0[0] * t)) + 0),
            f(f(SPAWN_Y + f(v0[1] * t)) + halfGravityT2),
            f(v0[2] * t),
        ];
        vel = [v0[0], f(v0[1] + f(-GRAVITY * t)), v0[2]];
    } else {
        vel = [vel[0], f(vel[1] + f(-GRAVITY * dt)), vel[2]];
        pos = [f(pos[0] + f(vel[0] * dt)), f(pos[1] + f(vel[1] * dt)), f(pos[2] + f(vel[2] * dt))];
        if (pos[1] <= 0 && vel[1] < 0) {
            vel = jetRef((i ^ Math.imul(frame, GOLDEN)) >>> 0);
            pos = [0, SPAWN_Y, 0];
        }
    }
    return { posSeed: [...pos, 1], vel: [...vel, 0] };
}

describe("fountain CPU logic", () => {
    test("u32 hash dual agrees with modulo arithmetic across the sign bit", () => {
        for (const seed of [0, 1, 2, 3, 0x01234567, 0x9e3779b9, 0xffffffff]) {
            expect(hashU32(seed), `seed ${seed}`).toBe(hashRef(seed));
        }
    });

    test("seeded, continuing, and recycle branches match the prior f32 integrator", () => {
        const cases: { i: number; pos: Vec4; vel: Vec4; dt: number; frame: number }[] = [
            { i: 3, pos: [0, 0, 0, 0], vel: [0, 0, 0, 0], dt: f(0.016), frame: 8 },
            { i: 97, pos: [0.2, 2.5, -0.1, 1], vel: [0.4, 1.2, -0.3, 0], dt: f(1 / 60), frame: 41 },
            {
                i: 50_000 - 1,
                pos: [0.1, -0.01, 0.2, 1],
                vel: [0, -2, 0, 0],
                dt: f(1 / 60),
                frame: 4_053_825_569,
            },
        ];
        for (const c of cases) {
            const actual = integrateParticle(c.i, d.vec4f(...c.pos), d.vec4f(...c.vel), {
                time: 1,
                dt: c.dt,
                frame: c.frame,
            });
            const expected = integrateRef(c.i, c.pos, c.vel, c.dt, c.frame);
            expect(Array.from(actual.posSeed)).toEqual(expected.posSeed);
            expect(Array.from(actual.vel)).toEqual(expected.vel);
        }
    });
});

describe("fountain emitted WGSL", () => {
    test("pins the exact helper graph and rejects a production-generated address mutation", () => {
        const kernel = flat(fountainIntegrateWgsl());
        integerDiscipline(kernel);
        const divisionAudited = kernel.replace(
            "((2f * v0.y) / 9.800000190734863f)",
            "flight-by-constant",
        );
        expect(divisionAudited).not.toBe(kernel);
        noDivision(divisionAudited);
        const hash = createHash("sha256").update(kernel).digest("hex");
        expect(hash).toBe("b5007b7c9a585a5f93b067df2ee703e3f71f263fa273cd0ed0b4137db08b2491");
        const mutated = flat(fountainIntegrateWgsl(true));
        expect(mutated).not.toBe(kernel);
        expect(createHash("sha256").update(mutated).digest("hex")).not.toBe(hash);
    });
});

type FakeState = {
    disposed: boolean;
    onDispose: (cleanup: () => void) => void;
    dispose: () => void;
};
type FakeBuffer = { kind: "production" | "warm"; generation: number; destroyed: number };
type FakeDraft = { state: FakeState; buffers: FakeBuffer[]; generation: number };
type FakeOwner = { state: FakeState; buffer: FakeBuffer; generation: number; cleaned: boolean };

function fakeState(): FakeState {
    const cleanups: (() => void)[] = [];
    const state: FakeState = {
        disposed: false,
        onDispose(cleanup) {
            cleanups.push(cleanup);
        },
        dispose() {
            if (state.disposed) return;
            state.disposed = true;
            for (let i = cleanups.length - 1; i >= 0; i--) cleanups[i]();
        },
    };
    return state;
}

function deferred() {
    let resolve!: () => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<void>((yes, no) => {
        resolve = yes;
        reject = no;
    });
    return { promise, resolve, reject };
}

function fakeLifecycle() {
    let active: FakeOwner | null = null;
    let generation = 0;
    let failPrepare = false;
    let rejectPrecompile = false;
    const owners = new WeakMap<FakeState, FakeOwner>();
    const drafts: FakeDraft[] = [];
    const created: FakeOwner[] = [];
    const published: number[] = [];
    const forced: FakeBuffer[] = [];
    const waits: ReturnType<typeof deferred>[] = [];
    const lifecycle = createFountainLifecycle<FakeState, FakeDraft, FakeOwner>({
        current: () => active,
        owned: (state) => owners.get(state),
        draft(state) {
            const draft = { state, buffers: [], generation: generation++ };
            drafts.push(draft);
            return draft;
        },
        prepare(draft) {
            const buffer: FakeBuffer = {
                kind: "production",
                generation: draft.generation,
                destroyed: 0,
            };
            draft.buffers.push(buffer);
            if (failPrepare) {
                failPrepare = false;
                throw new Error("partial allocation");
            }
            const owner = {
                state: draft.state,
                buffer,
                generation: draft.generation,
                cleaned: false,
            };
            created.push(owner);
            return owner;
        },
        activate(state, owner) {
            active = owner;
            owners.set(state, owner);
        },
        cleanupDraft(draft) {
            for (const buffer of draft.buffers) buffer.destroyed++;
        },
        cleanup(owner) {
            if (owner.cleaned) return;
            owner.cleaned = true;
            owner.buffer.destroyed++;
            if (owners.get(owner.state) === owner) owners.delete(owner.state);
            if (active === owner) active = null;
        },
        async precompile(owner, force) {
            if (rejectPrecompile) {
                rejectPrecompile = false;
                throw new Error("precompile failed");
            }
            const wait = deferred();
            waits.push(wait);
            await wait.promise;
            force();
            expect(active === owner || owner.cleaned).toBe(true);
        },
        force(owner) {
            const warm: FakeBuffer = {
                kind: "warm",
                generation: owner.generation,
                destroyed: 0,
            };
            forced.push(warm);
            warm.destroyed++;
            return warm;
        },
        publish(owner) {
            published.push(owner.generation);
        },
    });
    return {
        lifecycle,
        drafts,
        created,
        published,
        forced,
        waits,
        active: () => active,
        failPrepare: () => (failPrepare = true),
        rejectPrecompile: () => (rejectPrecompile = true),
    };
}

describe("fountain warm lifecycle", () => {
    test("keeps frame setup synchronous and assigns GPU preparation to warm", () => {
        expect(FountainSystem.setup).toBeUndefined();
        expect(FountainPlugin.warm).toBeDefined();
        expect(FountainPlugin.dispose).toBeDefined();
    });

    test("an already-disposed State remains allocation-free", async () => {
        const harness = fakeLifecycle();
        const state = fakeState();
        state.dispose();
        await harness.lifecycle.warm(state);
        expect(harness.drafts).toEqual([]);
        expect(harness.created).toEqual([]);
    });

    test("throwaway ownership cleans an earlier buffer when a later allocation throws", () => {
        const first = { destroyed: 0 };
        expect(() =>
            withOwnedFountainBuffers<typeof first, never>(
                (own) => {
                    own(first);
                    throw new Error("second allocation failed");
                },
                (buffer) => buffer.destroyed++,
            ),
        ).toThrow("second allocation failed");
        expect(first.destroyed).toBe(1);
    });

    test("old teardown cannot clear a replacement and every force owns throwaway bindings", async () => {
        const harness = fakeLifecycle();
        const oldState = fakeState();
        const nextState = fakeState();
        const oldWarm = harness.lifecycle.warm(oldState);
        await Promise.resolve();
        const nextWarm = harness.lifecycle.warm(nextState);
        await Promise.resolve();
        const [oldOwner, nextOwner] = harness.created;
        expect(oldOwner.cleaned).toBe(true);
        expect(harness.active()).toBe(nextOwner);

        harness.waits[0].resolve();
        await oldWarm;
        expect(harness.published).toEqual([]);
        oldState.dispose();
        expect(harness.active()).toBe(nextOwner);

        harness.waits[1].resolve();
        await nextWarm;
        expect(harness.published).toEqual([nextOwner.generation]);
        expect(harness.forced.map((buffer) => buffer.kind)).toEqual(["warm", "warm"]);
        expect(harness.forced.every((buffer) => buffer.destroyed === 1)).toBe(true);
        expect(harness.forced.every((buffer) => buffer !== nextOwner.buffer)).toBe(true);
        nextState.dispose();
        nextState.dispose();
        expect(nextOwner.buffer.destroyed).toBe(1);
    });

    test("exact publication cleanup preserves a replacement and is idempotent", () => {
        type Entry = { name: string; generation: number };
        type Gate = { generation: number };
        type Owner = {
            buffers: FakeBuffer[];
            particlesRaw: FakeBuffer;
            particles: { generation: number };
            draw: Entry;
            gate: Gate;
        };
        const makeOwner = (generation: number): Owner => {
            const particlesRaw: FakeBuffer = {
                kind: "production",
                generation,
                destroyed: 0,
            };
            return {
                buffers: [particlesRaw],
                particlesRaw,
                particles: { generation },
                draw: { name: "fountain", generation },
                gate: { generation },
            };
        };
        const old = makeOwner(1);
        const next = makeOwner(2);
        let raw: FakeBuffer | undefined = next.particlesRaw;
        let typed: { generation: number } | undefined = next.particles;
        let draw: Entry | undefined = next.draw;
        let gate: Gate | undefined = next.gate;
        let active: Owner | null = next;
        const ops = {
            raw: () => raw,
            typed: () => typed,
            draw: (name: string) => (draw?.name === name ? draw : undefined),
            gate: () => gate,
            deleteRaw: () => (raw = undefined),
            deleteTyped: () => (typed = undefined),
            deleteDraw: () => (draw = undefined),
            deleteGate: () => (gate = undefined),
            destroy: (buffer: FakeBuffer) => buffer.destroyed++,
            active: () => active,
            clearActive: () => (active = null),
        };

        cleanupFountainGeneration(old, ops);
        expect(old.particlesRaw.destroyed).toBe(1);
        expect(raw).toBe(next.particlesRaw);
        expect(typed).toBe(next.particles);
        expect(draw).toBe(next.draw);
        expect(gate).toBe(next.gate);
        expect(active).toBe(next);

        cleanupFountainGeneration(next, ops);
        cleanupFountainGeneration(next, ops);
        expect(next.particlesRaw.destroyed).toBe(1);
        expect(raw).toBeUndefined();
        expect(typed).toBeUndefined();
        expect(draw).toBeUndefined();
        expect(gate).toBeUndefined();
        expect(active).toBeNull();
    });

    test("partial preparation and precompile rejection clean their generation", async () => {
        const partial = fakeLifecycle();
        partial.failPrepare();
        await expect(partial.lifecycle.warm(fakeState())).rejects.toThrow("partial allocation");
        expect(partial.drafts[0].buffers[0].destroyed).toBe(1);
        expect(partial.active()).toBeNull();

        const rejected = fakeLifecycle();
        rejected.rejectPrecompile();
        await expect(rejected.lifecycle.warm(fakeState())).rejects.toThrow("precompile failed");
        expect(rejected.created[0].cleaned).toBe(true);
        expect(rejected.created[0].buffer.destroyed).toBe(1);
        expect(rejected.active()).toBeNull();
    });
});
