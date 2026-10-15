import { expect, spyOn, test } from "bun:test";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import {
    deriveTransforms,
    GlobalTransform,
    GlobalTransformTickStartSystem,
    PrepareGlobalTransformSystem,
    Transform,
    TransformRuntime,
} from "../../core/transform";
import { createApp, type Plugin, type System, Time } from "../../engine";
import { Character, CharacterPlugin } from "./character";
import { init, kernel } from "./kernel/kernel";
import {
    hashPhysics,
    physicsBodyIndex,
    physicsState,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    setTargetTransform,
    setTransform,
} from "./runtime";
import { bodyGetTransform } from "./world/body";

const SyncSystem = StandardPhysicsPlugin.systems![0];

function pose(world: Awaited<ReturnType<typeof createApp>>["world"], eid: number) {
    return bodyGetTransform(physicsState(world)!, physicsBodyIndex(world, eid), {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    });
}

test("a fixed Transform write after sync teleports before the same tick's solve", async () => {
    let eid = 0;
    let write = false;
    const writer: System = {
        name: "teleport-after-sync",
        group: "fixed",
        after: [SyncSystem],
        before: [StepPhysicsSystem],
        update(world) {
            if (write) world.storage(Transform).translation.set(eid, 7, 8, 9);
        },
    };
    const plugin: Plugin = {
        name: "Teleport",
        dependencies: [StandardPhysicsPlugin],
        recovery: "stateless",
        systems: [writer],
    };
    const app = await createApp({ defaults: false, plugins: [plugin] });
    try {
        const w = app.world;
        eid = w.create();
        w.add(eid, Body);
        w.tick();
        const setter = spyOn(kernel(w), "bodySetPose");
        try {
            w.resource(TransformRuntime).discontinuities.fill(0);
            write = true;
            w.tick();
            expect(
                w.resource(TransformRuntime).discontinuities[eid >>> 5] & (1 << (eid & 31)),
            ).toBe(0);
            expect(setter).toHaveBeenCalledTimes(1);
            expect(pose(w, eid).p).toEqual({ x: 7, y: 8, z: 9 });
            expect(w.storage(GlobalTransform).translation.x.get(eid)).toBe(7);
            w.tick();
            expect(setter).toHaveBeenCalledTimes(1);
        } finally {
            setter.mockRestore();
        }
    } finally {
        app.dispose();
    }
});

test("unchanged moving poses and snapshot restore issue no set-transform", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const w = app.world,
            eid = w.create();
        w.add(eid, Transform, { translation: [2, 10, 3], rotation: [0, 0, 0.6, 0.8] });
        w.add(eid, Body, { type: BodyType.Dynamic, angularVelocity: [1, 2, 3] });
        w.add(eid, Shape);
        w.tick();
        const saved = w.snapshot(),
            before = hashPhysics(w);
        const setter = spyOn(kernel(w), "bodySetPose");
        try {
            for (let tick = 0; tick < 4; tick++) w.tick();
            w.restore(saved);
            expect(hashPhysics(w)).toBe(before);
            const p = pose(w, eid);
            const t = w.storage(Transform);
            expect([
                t.translation.x.get(eid),
                t.translation.y.get(eid),
                t.translation.z.get(eid),
            ]).toEqual([p.p.x, p.p.y, p.p.z]);
            expect([
                t.rotation.x.get(eid),
                t.rotation.y.get(eid),
                t.rotation.z.get(eid),
                t.rotation.w.get(eid),
            ]).toEqual([p.q.v.x, p.q.v.y, p.q.v.z, p.q.s]);
            w.tick();
            expect(setter).not.toHaveBeenCalled();
        } finally {
            setter.mockRestore();
        }
    } finally {
        app.dispose();
    }
});

test("a post-step target survives tick-end, tick-start and draw derivation without teleporting", async () => {
    let eid = 0,
        target = false;
    const publish: System = {
        name: "target-after-step",
        group: "fixed",
        after: [StepPhysicsSystem],
        update(world) {
            if (target) setTargetTransform(world, eid, [6, 0, 0], [0, 0, 0, 1], 1);
        },
    };
    const plugin: Plugin = {
        name: "Target",
        dependencies: [StandardPhysicsPlugin],
        recovery: "stateless",
        systems: [publish],
    };
    const app = await createApp({ defaults: false, plugins: [plugin] });
    try {
        const w = app.world;
        eid = w.create();
        w.add(eid, Body, { type: BodyType.Kinematic });
        w.tick();
        const setter = spyOn(kernel(w), "bodySetPose");
        try {
            target = true;
            w.tick();
            target = false;
            const global = w.storage(GlobalTransform),
                authored = w.storage(Transform);
            expect(global.translation.x.get(eid)).toBe(6);
            expect(authored.translation.x.get(eid)).toBe(0);
            expect(pose(w, eid).p.x).toBe(0);
            GlobalTransformTickStartSystem.update!(w);
            expect(global.translation.x.get(eid)).toBe(6);
            PrepareGlobalTransformSystem.update!(w);
            expect(global.translation.x.get(eid)).toBe(6);
            w.step(0);
            expect(global.translation.x.get(eid)).toBe(6);
            w.tick();
            expect(pose(w, eid).p.x).toBeCloseTo(6 * Time.FIXED_DT, 6);
            expect(authored.translation.x.get(eid)).toBe(pose(w, eid).p.x);
            expect(setter).not.toHaveBeenCalled();
        } finally {
            setter.mockRestore();
        }
    } finally {
        app.dispose();
    }
});

test("a kinematic target retains its trajectory without set-transform", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "OneThread",
                dependencies: [StandardPhysicsPlugin],
                recovery: "stateless",
                initialize: (world) => init(world, { threads: 1 }),
            },
        ],
    });
    try {
        const w = app.world,
            eid = w.create();
        w.add(eid, Body, { type: BodyType.Kinematic });
        w.add(eid, Shape);
        w.tick();
        const setter = spyOn(kernel(w), "bodySetPose");
        try {
            setTargetTransform(w, eid, [6, 3, -2], [0, 0, 0, 1], 1);
            const hashes: bigint[] = [];
            for (let tick = 0; tick < 8; tick++) {
                w.tick();
                hashes.push(hashPhysics(w));
                expect(w.storage(Transform).translation.x.get(eid)).toBe(pose(w, eid).p.x);
            }
            expect(hashes).toEqual([
                5963752036176916748n,
                479078990755557644n,
                18156159298392922985n,
                1548134450822803724n,
                3780868718480870772n,
                8758850759908084522n,
                10538555932609507376n,
                2232927688150771406n,
            ]);
            expect(setter).not.toHaveBeenCalled();
        } finally {
            setter.mockRestore();
        }
    } finally {
        app.dispose();
    }
});

test("body pose claims retain scale derivation and release across removal, recycling and restore", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const w = app.world,
            eid = w.create(),
            ordinary = w.create();
        w.add(eid, Body);
        w.add(ordinary, Transform);
        w.tick();
        const saved = w.snapshot();
        const t = w.storage(Transform),
            g = w.storage(GlobalTransform);
        t.translation.set(eid, 20, 0, 0);
        t.scale.set(eid, 2, 3, 4);
        t.translation.set(ordinary, 30, 0, 0);
        deriveTransforms(w);
        expect(g.translation.x.get(eid)).toBe(0);
        expect(g.scale.y.get(eid)).toBe(3);
        expect(g.translation.x.get(ordinary)).toBe(30);
        w.remove(eid, Body);
        deriveTransforms(w);
        expect(g.translation.x.get(eid)).toBe(20);
        w.restore(saved);
        t.translation.set(eid, 40, 0, 0);
        deriveTransforms(w);
        expect(g.translation.x.get(eid)).toBe(0);
        w.destroy(eid);
        expect(w.create()).toBe(eid);
        w.add(eid, Transform, { translation: [50, 0, 0] });
        deriveTransforms(w);
        expect(g.translation.x.get(eid)).toBe(50);
        expect(w.resource(TransformRuntime).poseClaims[eid >>> 5] & (1 << (eid & 31))).toBe(0);
    } finally {
        app.dispose();
    }
});

test("a Character teleports before moving and its endpoint makes the top-of-step compare a no-op", async () => {
    const app = await createApp({ defaults: false, plugins: [CharacterPlugin] });
    try {
        const w = app.world,
            eid = w.create();
        w.add(eid, Body, { type: BodyType.Kinematic });
        w.add(eid, Shape, { kind: ShapeKind.Capsule });
        w.add(eid, Character);
        w.tick();
        const setter = spyOn(kernel(w), "bodySetPose");
        try {
            w.storage(Transform).translation.set(eid, 50, 10, 0);
            w.storage(Character).velocity.set(eid, 3, 0, 0);
            w.tick();
            const x = w.storage(Transform).translation.x.get(eid);
            expect(x).toBeCloseTo(50 + 3 * Time.FIXED_DT, 5);
            expect(pose(w, eid).p.x).toBe(x);
            expect(w.storage(GlobalTransform).translation.x.get(eid)).toBe(x);
            // One authored teleport, one character endpoint; no third call at the solver boundary.
            expect(setter).toHaveBeenCalledTimes(2);
            w.tick();
            expect(setter).toHaveBeenCalledTimes(3);
        } finally {
            setter.mockRestore();
        }
    } finally {
        app.dispose();
    }
});

test("a pending Transform teleport survives draw-only mark clearing and recovery", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const w = app.world,
            eid = w.create();
        w.add(eid, Body);
        w.tick();
        w.storage(Transform).translation.set(eid, 12, 3, 4);
        w.step(0);
        const saved = w.snapshot();
        w.tick();
        expect(pose(w, eid).p.x).toBe(12);
        w.restore(saved);
        const setter = spyOn(kernel(w), "bodySetPose");
        try {
            w.tick();
            expect(pose(w, eid).p.x).toBe(12);
            expect(setter).toHaveBeenCalledTimes(1);
            w.tick();
            expect(setter).toHaveBeenCalledTimes(1);
        } finally {
            setter.mockRestore();
        }
    } finally {
        app.dispose();
    }
});

test("setTransform updates both pose columns without a second teleport", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const w = app.world,
            eid = w.create();
        w.add(eid, Body);
        w.tick();
        const setter = spyOn(kernel(w), "bodySetPose");
        try {
            setTransform(w, eid, [5, 6, 7], [0, 0, 1, 0]);
            expect(w.storage(Transform).translation.x.get(eid)).toBe(5);
            expect(w.storage(GlobalTransform).rotation.z.get(eid)).toBe(1);
            w.tick();
            expect(setter).toHaveBeenCalledTimes(1);
        } finally {
            setter.mockRestore();
        }
    } finally {
        app.dispose();
    }
});
