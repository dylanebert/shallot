import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { type App, createApp, type World } from "../../engine";
import { type ClientId, type Connection, type Game, local, ServerSession } from ".";
import { ServerMessageKind } from "./server-message";

const emptyGame: Game = {
    plugins: [],
    schema: [],
    maxClients: 2,
    rules: () => [],
    validateIntent: () => false,
    applyIntent() {},
};
const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };

function messageTopics(packets: Uint8Array[]): number[] {
    const topics: number[] = [];
    for (const bytes of packets) {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        if (view.getUint32(0, true) === ServerMessageKind) topics.push(view.getUint32(4, true));
    }
    return topics;
}

test("a departed private target never reaches the replacement reusing its eid", async () => {
    const joined: { eid: number; id: ClientId }[] = [];
    let departed: ClientId | undefined;
    let authority!: World;
    let wire!: Connection;
    const host = await local(
        {
            ...emptyGame,
            onJoin(world, eid, _transport, id) {
                authority = world;
                joined.push({ eid, id });
            },
            onLeave(_world, _eid, id) {
                departed = id;
            },
            onClient(_world, connection) {
                wire = connection;
            },
        },
        { rooms: true, seed: 1, faults },
    );
    let app: App | undefined;
    try {
        const first = await host.join();
        await host.join(first.code);
        const server = authority.resource(ServerSession).server!;
        const oldTarget = joined[0]!.id;
        first.leave();
        expect(departed).toBe(oldTarget);
        // Wrap the ECS generation without admitting any new seats.
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            for (let i = 0; i < 2 ** 21 - 1; i++) authority.destroy(authority.create());
            expect(warn).toHaveBeenCalledTimes(1);
        } finally {
            warn.mockRestore();
        }
        const replacement = await host.join(first.code);
        expect(joined[2]!.eid).toBe(joined[0]!.eid);
        app = await createApp({ defaults: false, plugins: [replacement.plugin] });
        app.world.step(0);
        server.sendMessage(oldTarget, { topic: 1, values: [123] });
        server.sendMessage(joined[2]!.id, { topic: 2, values: [] });
        host.pump(0);
        expect(messageTopics(wire.receive())).toEqual([2]);
    } finally {
        app?.dispose();
        host.dispose();
    }
});
