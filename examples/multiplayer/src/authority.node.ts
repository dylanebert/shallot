import { expect, setDefaultTimeout, test } from "bun:test";
import { GlobalTransform } from "@dylanebert/shallot";
import { IntentSender } from "@dylanebert/shallot/multiplayer";
import { createSession } from "@dylanebert/shallot/multiplayer/testing";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { GAME, scene } from "./network";
import { PlayerInput } from "./player";

setDefaultTimeout(TIMEOUT.node);
const invalid = [
    ["extra target entity", [0, 1, 0, 0, 0, 0, 4]],
    ["missing field", [0, 1]],
    ["nonfinite movement", [0, NaN, 0, 0, 0, 0]],
    ["nonfinite yaw", [0, 1, Infinity, 0, 0, 0]],
    ["lateral speed", [2, 0, 0, 0, 0, 0]],
    ["forward speed", [0, 2, 0, 0, 0, 0]],
    ["unbounded yaw", [0, 1, 1e7, 0, 0, 0]],
    ["pitch", [0, 1, 0, Math.PI, 0, 0]],
    ["sprint", [0, 1, 0, 0, 2, 0]],
    ["fractional jump", [0, 1, 0, 0, 0, 0.5]],
    ["negative jump", [0, 1, 0, 0, 0, -1]],
    ["overflowed jump", [0, 1, 0, 0, 0, 2 ** 32]],
] as const;
for (const [name, values] of invalid) {
    test(`server refuses intent with ${name}`, async () => {
        const session = await createSession(GAME, { seed: 5, clients: 1 });
        const app = session.server;
        try {
            const input = new IntentSender(session.clients[0]!.connection!);
            input.record(1, "input", [...values]);
            input.flush();
            await session.step(session.server);
            const player = scene(app.world).players[0]!.player;
            expect(app.world.get(player, PlayerInput)?.move ?? [0, 0]).toEqual([0, 0]);
            expect(app.world.get(player, PlayerInput)?.sprint ?? 0).toBe(0);
            expect(app.world.get(player, PlayerInput)?.jumpPresses ?? 0).toBe(0);
            expect(app.world.storage(GlobalTransform).translation.x.get(player)).toBe(2);
            expect(app.world.storage(GlobalTransform).translation.z.get(player)).toBe(15);
        } finally {
            await session.dispose();
        }
    });
}

test("the connection selects the avatar; operations and requested future ticks cannot move another player", async () => {
    const session = await createSession(GAME, { seed: 1, clients: 2 });
    const app = session.server;
    try {
        const input = new IntentSender(session.clients[0]!.connection!);
        input.record(1, "operation", [0, 1, 0, 0, 0, 0]);
        input.flush();
        await session.step(session.server);
        const [own, other] = scene(app.world).players;
        const pose = app.world.storage(GlobalTransform).translation;
        expect(pose.z.get(own!.player)).toBe(15);
        input.record(10, "input", [0, 1, 0, 0, 0, 0]);
        input.flush();
        await session.step(session.server, 8);
        expect(pose.z.get(own!.player)).toBe(15);
        await session.step(session.server);
        expect(pose.z.get(own!.player)).toBeLessThan(15);
        expect(pose.x.get(other!.player)).toBe(-2);
        expect(pose.z.get(other!.player)).toBe(10);
        await session.step(session.server, 30);
        expect(pose.z.get(other!.player)).toBe(10);
    } finally {
        await session.dispose();
    }
});
