import { expect, setDefaultTimeout, test } from "bun:test";
import { Character, GlobalTransform, GroundState, setTransform } from "@dylanebert/shallot";
import { ServerSession, seededLink } from "@dylanebert/shallot/multiplayer";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { GameClient, openGame, openServer, scene } from "./network";

setDefaultTimeout(TIMEOUT.node);
test("a predicted rider stays on the moving lift at 100 ms and 2% loss", async () => {
    const apps = [await openServer(), await openGame(), await openGame()];
    try {
        for (const app of apps)
            setTransform(
                app.world,
                scene(app.world).players[0]!.player,
                [0, 2.95, -6.5],
                [0, 0, 0, 1],
            );
        const server = apps[0]!.world.resource(ServerSession).server!;
        const links = [
            seededLink(1, { minDelay: 50, maxDelay: 50, loss: 0.02, duplication: 0 }),
            seededLink(2, { minDelay: 50, maxDelay: 50, loss: 0.02, duplication: 0 }),
        ];
        const clients = links.map((link, index) => {
            const client = new GameClient(apps[index + 1]!.world, link.b);
            server.connect(link.a);
            return client;
        });
        let maximumGapError = 0,
            minimumY = Infinity,
            maximumY = -Infinity;
        for (let frame = 0; frame < 240; frame++) {
            for (const link of links) link.advance((frame * 1000) / 60);
            for (const client of clients) {
                client.record([0, 0, 0, 0, 0, 0]);
                client.step(1 / 60);
            }
            server.world.tick();
            if (frame < 30) continue;
            const rider = scene(clients[0]!.world),
                route = scene(server.world);
            const clientY = clients[0]!.world.storage(GlobalTransform).translation.y;
            const serverY = server.world.storage(GlobalTransform).translation.y;
            const gap = clientY.get(rider.players[0]!.player) - clientY.get(rider.lift);
            const authoritativeGap =
                serverY.get(route.players[0]!.player) - serverY.get(route.lift);
            maximumGapError = Math.max(maximumGapError, Math.abs(gap - authoritativeGap));
            minimumY = Math.min(minimumY, clientY.get(rider.players[0]!.player));
            maximumY = Math.max(maximumY, clientY.get(rider.players[0]!.player));
        }
        expect(maximumY - minimumY).toBeGreaterThan(1);
        expect(maximumGapError).toBeLessThan(0.08);
        const rider = scene(clients[0]!.world).players[0]!.player;
        expect(clients[0]!.world.storage(Character).groundState.get(rider)).toBe(
            GroundState.OnGround,
        );
    } finally {
        for (const app of apps) app.dispose();
    }
});
