import { expect, test } from "bun:test";
import { createApp, type World } from "../../engine";
import { type Game, local } from ".";

test("local composes isolated server, thin and caller-owned client Worlds and disposes its children with the client", async () => {
    const worlds: World[] = [],
        clients: World[] = [];
    const game: Game = {
        plugins: [
            {
                name: "RoleTest",
                initialize(world) {
                    worlds.push(world);
                },
            },
        ],
        schema: [],
        maxClients: 2,
        rules: () => [],
        validateIntent: () => true,
        applyIntent() {},
        onClient(world) {
            world.snapshot();
            clients.push(world);
        },
    };
    const session = await local(game, {
        seed: 1,
        faults: { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 },
        thinClients: [{ seed: 2, input() {} }],
    });
    const app = await createApp({ defaults: false, plugins: [...game.plugins, session.plugin] });
    try {
        expect(worlds).toHaveLength(3);
        expect(new Set(worlds).size).toBe(3);
        expect(worlds[2]).toBe(app.world);
        expect(clients).toEqual([]);
        app.world.step(0);
        app.world.step(0);
        expect(clients).toEqual([app.world]);
        session.pump(1 / 60);
        expect(worlds[0]!.time.fixedTick).toBe(1);
        expect(worlds[1]!.time.fixedTick).toBe(0);
        app.dispose();
        expect(worlds.every((world) => world.signal.aborted)).toBe(true);
    } finally {
        app.dispose();
        session.dispose();
    }
});
