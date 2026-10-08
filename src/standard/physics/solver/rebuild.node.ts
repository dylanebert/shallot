import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../../scripts/test-tiers";
import { PhysicsWorld } from "../api/world";
import { contactCount } from "../collision/contact";
import { BodyType } from "../common/types";
import { init, shutdown, threads } from "../kernel/kernel";

setDefaultTimeout(CEILING.node);

if (process.env.PHYSICS_REBUILD_BULLET) {
    const count = Number(process.env.PHYSICS_REBUILD_BULLET);
    await init(undefined, { threads: count });
    expect(threads(undefined)).toBe(count);
    afterAll(() => shutdown(undefined));
    test("a bullet hits a dynamic body after moving proxies rebuild the dynamic tree", () => {
        const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
        try {
            let seed = 42;
            const random = () => {
                seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
                return seed / 4294967296;
            };
            const bodies = [];
            for (let i = 0; i < 40; ++i) {
                const body = world.createBody({
                    type: BodyType.Dynamic,
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
                });
                body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
                body.applyMassFromShapes();
                bodies.push(body);
            }
            for (let i = 0; i < 8; ++i) {
                world.step(1 / 60, 1);
                expect(contactCount(world.state)).toBeLessThan(256 * 3);
            }
            const target = bodies[2];
            target.setLinearVelocity({ x: 0, y: 0, z: 0 });
            const position = target.getPosition();
            const bullet = world.createBody({
                type: BodyType.Dynamic,
                isBullet: true,
                position: { x: position.x - 5, y: position.y, z: position.z },
                linearVelocity: { x: 600, y: 0, z: 0 },
            });
            bullet.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.2 });
            bullet.applyMassFromShapes();
            const root = world.state.broadPhase.trees[BodyType.Dynamic].root;
            const poses = [];
            for (let i = 0; i < 3; ++i) {
                world.step(1 / 60, 1);
                expect(contactCount(world.state)).toBeLessThan(256 * 3);
                if (i === 0)
                    expect(world.state.broadPhase.trees[BodyType.Dynamic].root).not.toBe(root);
                poses.push(bullet.getPosition(), target.getPosition());
            }
            console.log(`BULLET_POSES=${JSON.stringify(poses)}`);
        } finally {
            world.destroy();
        }
    });
} else if (process.env.PHYSICS_REBUILD_POOL !== "1") {
    test("four-thread rebuild paths finish in an independent kernel runtime", async () => {
        const child = Bun.spawn([process.execPath, "test", import.meta.filename], {
            env: { ...process.env, PHYSICS_REBUILD_POOL: "1" },
            stdout: "inherit",
            stderr: "inherit",
        });
        expect(await child.exited).toBe(0);
    });
    test("an unforked four-thread bullet sweep uses the rebuilt dynamic root and equals one thread", async () => {
        const results = [];
        for (const count of [1, 4]) {
            const child = Bun.spawn([process.execPath, "test", import.meta.filename], {
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
        expect(results[0][0].x).toBeLessThan(results[0][1].x);
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
            const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
            try {
                for (let i = 0; i < count; ++i) {
                    const body = world.createBody({
                        type,
                        isAwake: type === BodyType.Dynamic,
                        linearVelocity: { x: type === BodyType.Dynamic ? 1 : 0, y: 0, z: 0 },
                    });
                    body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
                    body.applyMassFromShapes();
                    expect(body.isAwake()).toBe(type === BodyType.Dynamic);
                }
                world.step(dt, 1);
                if (name === "forked collide")
                    expect(contactCount(world.state)).toBeGreaterThan(256 * 3);
                if (name === "unforked collide with moving bodies") {
                    expect(contactCount(world.state)).toBeGreaterThan(0);
                    expect(contactCount(world.state)).toBeLessThan(256 * 3);
                    expect(world.getBodyEvents().count).toBe(2);
                }
            } finally {
                world.destroy();
            }
        });
    }
}
