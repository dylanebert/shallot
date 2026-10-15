import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { bunChild } from "../../../../scripts/bun-child";
import { contactCount } from "../collision/contact";
import type { WorldTransform } from "../common/records";
import { type BodyDef, BodyType } from "../common/types";
import { EventKind, eventCount } from "../kernel/eventbuffers";
import { init, shutdown, threads } from "../kernel/kernel";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import {
    bodyGetTransform,
    bodyIsAwake,
    bodySetLinearVelocity,
    updateBodyMassData,
} from "../world/body";
import type { WorldState } from "../world/world";
import { step as stepWorld } from "./step";

setDefaultTimeout(TIMEOUT.node);

const zero = { x: 0, y: 0, z: 0 };
const sphere = { center: zero, radius: 0.5 };
function bodyTransform(state: WorldState, body: number) {
    const out: WorldTransform = { p: { x: 0, y: 0, z: 0 }, q: { v: zero, s: 1 } };
    bodyGetTransform(state, body, out);
    return { p: { ...out.p }, q: { v: { ...out.q.v }, s: out.q.s } };
}
function addSphere(
    state: WorldState,
    type: BodyType,
    def: Partial<BodyDef> = {},
    radius = 0.5,
): number {
    const body = createTestBody(state, { type, ...def });
    if (createSphereShape(state, body, testShapeDef(), { ...sphere, radius }) === null)
        throw new Error("rebuild sphere creation failed");
    updateBodyMassData(state, body);
    return body;
}

if (process.env.PHYSICS_REBUILD_BULLET) {
    const count = Number(process.env.PHYSICS_REBUILD_BULLET);
    await init(undefined, { threads: count });
    expect(threads(undefined)).toBe(count);
    afterAll(() => shutdown(undefined));
    test("a bullet hits a dynamic body after moving proxies rebuild the dynamic tree", () => {
        const state = createTestWorld({ gravity: zero, enableSleep: false });
        try {
            let seed = 42;
            const random = () => {
                seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
                return seed / 4294967296;
            };
            const bodies = [];
            for (let i = 0; i < 40; ++i) {
                bodies.push(
                    addSphere(state, BodyType.Dynamic, {
                        position: {
                            x: random() * 20 - 10,
                            y: random() * 20 - 10,
                            z: random() * 20 - 10,
                        },
                        linearVelocity: {
                            x: random() * 10 - 5,
                            y: random() * 10 - 5,
                            z: random() * 10 - 5,
                        },
                    }),
                );
            }
            for (let i = 0; i < 8; ++i) {
                stepWorld(state, 1 / 60, 1);
                expect(contactCount(state)).toBeLessThan(256 * 3);
            }
            const target = bodies[2]!;
            bodySetLinearVelocity(state, target, zero);
            const position = bodyTransform(state, target).p;
            const bullet = addSphere(
                state,
                BodyType.Dynamic,
                {
                    isBullet: true,
                    position: { x: position.x - 5, y: position.y, z: position.z },
                    linearVelocity: { x: 600, y: 0, z: 0 },
                },
                0.2,
            );
            const root = state.broadPhase.trees[BodyType.Dynamic]!.root;
            const poses = [];
            for (let i = 0; i < 3; ++i) {
                stepWorld(state, 1 / 60, 1);
                expect(contactCount(state)).toBeLessThan(256 * 3);
                if (i === 0) expect(state.broadPhase.trees[BodyType.Dynamic]!.root).not.toBe(root);
                poses.push(bodyTransform(state, bullet), bodyTransform(state, target));
            }
            console.log(`BULLET_POSES=${JSON.stringify(poses)}`);
        } finally {
            destroyTestWorld(state);
        }
    });
} else if (process.env.PHYSICS_REBUILD_POOL !== "1") {
    test("four-thread rebuild paths finish in an independent kernel runtime", async () => {
        const child = Bun.spawn({
            ...bunChild(["test", import.meta.filename]),
            env: { ...process.env, PHYSICS_REBUILD_POOL: "1" },
            stdout: "inherit",
            stderr: "inherit",
        });
        expect(await child.exited).toBe(0);
    });
    test("an unforked four-thread bullet sweep uses the rebuilt dynamic root and equals one thread", async () => {
        const results = [];
        for (const count of [1, 4]) {
            const child = Bun.spawn({
                ...bunChild(["test", import.meta.filename]),
                env: { ...process.env, PHYSICS_REBUILD_BULLET: String(count) },
                stdout: "pipe",
                stderr: "inherit",
            });
            const output = await new Response(child.stdout).text();
            expect(await child.exited).toBe(0);
            const record = output.split("\n").find((line) => line.startsWith("BULLET_POSES="));
            if (!record) throw new Error("bullet child did not publish poses");
            results.push(JSON.parse(record.slice("BULLET_POSES=".length)));
        }
        expect(results[0][0].p.x).toBeLessThan(results[0][1].p.x);
        expect(results[1]).toEqual(results[0]);
    });
} else {
    await init(undefined, { threads: 4 });
    expect(threads(undefined)).toBe(4);
    afterAll(() => shutdown(undefined));

    for (const [name, count, dt, type] of [
        ["forked collide", 60, 1 / 60, BodyType.Dynamic],
        ["unforked collide with moving bodies", 2, 1 / 60, BodyType.Dynamic],
        ["zero-dt step", 2, 0, BodyType.Dynamic],
        ["step with no awake bodies", 2, 1 / 60, BodyType.Kinematic],
    ] as const) {
        test(`trees finish rebuilding before refit or the end of a four-thread ${name}`, () => {
            expect(threads(undefined)).toBe(4);
            const state = createTestWorld({ gravity: zero });
            try {
                for (let i = 0; i < count; ++i) {
                    const body = addSphere(state, type, {
                        isAwake: type === BodyType.Dynamic,
                        linearVelocity: { x: type === BodyType.Dynamic ? 1 : 0, y: 0, z: 0 },
                    });
                    expect(bodyIsAwake(state, body)).toBe(type === BodyType.Dynamic);
                }
                stepWorld(state, dt, 1);
                if (name === "forked collide") expect(contactCount(state)).toBeGreaterThan(256 * 3);
                if (name === "unforked collide with moving bodies") {
                    expect(contactCount(state)).toBeGreaterThan(0);
                    expect(contactCount(state)).toBeLessThan(256 * 3);
                    expect(eventCount(state, EventKind.BodyMove)).toBe(2);
                }
            } finally {
                destroyTestWorld(state);
            }
        });
    }
}
