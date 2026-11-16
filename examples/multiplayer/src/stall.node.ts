import { expect, setDefaultTimeout, test } from "bun:test";
import { createSession } from "@dylanebert/shallot/multiplayer/testing";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { ClientSession, GAME } from "./network";

setDefaultTimeout(TIMEOUT.node);
test("the socket session resumes confirmation within 40 ticks after a 20-tick downstream stall", async () => {
    const session = await createSession(GAME, {
        clients: 2,
        transport: "websocket",
        proxy: true,
        seed: 61,
    });
    try {
        await session.advance(10);
        const peer = session.clients[0]!;
        const client = peer.world.resource(ClientSession).client!;
        const receiver = client.prediction.receiver;
        // Drain in-flight bytes before taking the held confirmation baseline.
        peer.proxy!.downstream.stalled = true;
        await session.advance(4);
        const before = receiver.confirmed;
        for (let tick = 0; tick < 20; tick++) {
            for (const other of session.clients)
                other.world.resource(ClientSession).client!.record([0, 0, 0, 0, 0, 0]);
            await session.advance();
        }
        expect(receiver.confirmed).toBe(before);
        const target = session.server.world.time.fixedTick;
        expect(target - before).toBeGreaterThanOrEqual(20);
        peer.proxy!.downstream.stalled = false;
        await session.waitFor(
            peer,
            () => receiver.confirmed >= target,
            40,
            "confirmation after socket stall",
        );
        expect(peer.connection!.closed).toBe(false);
        expect(client.slot).toBe(0);
    } finally {
        await session.dispose();
    }
});
