import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, Character, GlobalTransform } from "@dylanebert/shallot";
import { InterpolationTarget, ServerSession, seededLink } from "@dylanebert/shallot/multiplayer";
import { readRenderGlobalTransform } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { GameClient, openGame, openServer, scene } from "./network";
import { Player, PlayerInput } from "./player";

setDefaultTimeout(TIMEOUT.node);
for (const simulated of [false, true]) {
    test(`a server and two clients move separate avatars and push the shared server-owned box${simulated ? " at 100 ms and 2% loss" : ""}`, async () => {
        const apps = [await openServer(), await openGame(), await openGame()];
        try {
            const server = apps[0]!.world.resource(ServerSession).server!;
            const links = [
                seededLink(10, {
                    minDelay: simulated ? 50 : 0,
                    maxDelay: simulated ? 50 : 0,
                    loss: simulated ? 0.02 : 0,
                    duplication: 0,
                }),
                seededLink(11, {
                    minDelay: simulated ? 50 : 0,
                    maxDelay: simulated ? 50 : 0,
                    loss: simulated ? 0.02 : 0,
                    duplication: 0,
                }),
            ];
            const clients = links.map((link, i) => {
                const client = new GameClient(apps[i + 1]!.world, link.b);
                server.connect(link.a);
                return client;
            });
            const route = scene(server.world);
            for (let i = 0; i < 150; i++) {
                for (const link of links) link.advance((i * 1000) / 60);
                for (const [slot, client] of clients.entries()) {
                    client.record([slot === 0 ? 0 : 1, slot === 0 ? 1 : 0, 0, 0, 0, 0]);
                    client.step(1 / 60);
                }
                server.world.tick();
            }
            const pos = server.world.storage(GlobalTransform).translation;
            expect(pos.z.get(route.players[0]!.player)).toBeLessThan(13);
            expect(pos.x.get(route.players[1]!.player)).toBeGreaterThan(0);
            expect(pos.z.get(route.box)).toBeLessThan(11);
            expect(clients.map((c) => c.slot)).toEqual([0, 1]);
            expect(clients[0]!.prediction.predicted(scene(clients[0]!.world).box)).toBe(true);
            expect(clients[1]!.prediction.predicted(scene(clients[1]!.world).box)).toBe(false);
            expect(server.world.storage(Body).type.get(route.box)).toBe(BodyType.Dynamic);
            for (const [slot, client] of clients.entries()) {
                const own = scene(client.world).players[slot]!.player;
                const remote = scene(client.world).players[1 - slot]!.player;
                expect(client.prediction.predicted(own)).toBe(true);
                expect(client.prediction.predicted(remote)).toBe(false);
                expect(client.world.has(remote, InterpolationTarget)).toBe(true);
                if (slot === 0) {
                    const displayed = new Float32Array(12);
                    readRenderGlobalTransform(client.world, remote, displayed);
                    const confirmed = client.world
                        .storage(GlobalTransform)
                        .translation.x.get(remote);
                    expect(displayed[0]).toBeLessThan(confirmed - 0.01);
                }
                expect(client.world.has(own, Character)).toBe(true);
                for (const component of [Character, Player, PlayerInput])
                    expect(client.world.has(remote, component)).toBe(false);
            }
        } finally {
            for (const app of apps) app.dispose();
        }
    });
}
