import { expect, setDefaultTimeout, test } from "bun:test";
import { GlobalTransform } from "@dylanebert/shallot";
import { IntentSender, ServerSession, seededLink } from "@dylanebert/shallot/multiplayer";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openServer, scene } from "./network";
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
        const app = await openServer();
        try {
            const server = app.world.resource(ServerSession).server!;
            const link = seededLink(5, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
            server.connect(link.a);
            const input = new IntentSender(link.b);
            input.record(1, "input", [...values]);
            input.flush();
            server.world.tick();
            const player = scene(app.world).players[0]!.player;
            expect(app.world.get(player, PlayerInput)?.move ?? [0, 0]).toEqual([0, 0]);
            expect(app.world.get(player, PlayerInput)?.sprint ?? 0).toBe(0);
            expect(app.world.get(player, PlayerInput)?.jumpPresses ?? 0).toBe(0);
            expect(app.world.storage(GlobalTransform).translation.x.get(player)).toBe(2);
            expect(app.world.storage(GlobalTransform).translation.z.get(player)).toBe(15);
        } finally {
            app.dispose();
        }
    });
}

test("the connection selects the avatar; operations and requested future ticks cannot move another player", async () => {
    const app = await openServer();
    try {
        const server = app.world.resource(ServerSession).server!;
        const links = [
            seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 }),
            seededLink(2, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 }),
        ];
        for (const link of links) server.connect(link.a);
        const input = new IntentSender(links[0]!.b);
        input.record(1, "operation", [0, 1, 0, 0, 0, 0]);
        input.flush();
        server.world.tick();
        const [own, other] = scene(app.world).players;
        const pose = app.world.storage(GlobalTransform).translation;
        expect(pose.z.get(own!.player)).toBe(15);
        input.record(10, "input", [0, 1, 0, 0, 0, 0]);
        input.flush();
        for (let i = 0; i < 8; i++) server.world.tick();
        expect(pose.z.get(own!.player)).toBe(15);
        server.world.tick();
        expect(pose.z.get(own!.player)).toBeLessThan(15);
        expect(pose.x.get(other!.player)).toBe(-2);
        expect(pose.z.get(other!.player)).toBe(10);
        for (let i = 0; i < 30; i++) server.world.tick();
        expect(pose.z.get(other!.player)).toBe(10);
    } finally {
        app.dispose();
    }
});
