import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, Time } from "@dylanebert/shallot";
import { client, replaySession, type SessionRecording } from "@dylanebert/shallot/multiplayer";
import { serveGame } from "@dylanebert/shallot/multiplayer/bun";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { ClientSession, GAME } from "./network";

setDefaultTimeout(TIMEOUT.node);
test("the example's live Bun server recording replays headlessly with identical fixed-tick hashes", async () => {
    let recording: SessionRecording | undefined;
    const endpoint = await serveGame(GAME, {
        hostname: "127.0.0.1",
        port: 0,
        recording: {
            gameVersion: "example-1",
            write: (value) => {
                recording = value;
            },
        },
    });
    const app = await createApp({
        defaults: false,
        plugins: [client(GAME, { url: `ws://127.0.0.1:${endpoint.port}` })],
    });
    try {
        for (let i = 0; endpoint.world.time.fixedTick < 40 && i < 2000; i++) {
            app.world.resource(ClientSession).client?.record([0, 1, 0, 0, 0, 0]);
            app.world.step(Time.FIXED_DT);
            await Bun.sleep(5);
        }
        expect(endpoint.world.time.fixedTick).toBeGreaterThanOrEqual(40);
    } finally {
        app.dispose();
        endpoint.stop();
    }
    expect(recording).toBeDefined();
    const hashes = (calls: SessionRecording["calls"]): number =>
        calls.reduce((count, call) => count + Number(call.kind === "hash") + hashes(call.calls), 0);
    expect(hashes(recording!.calls)).toBe(endpoint.world.time.fixedTick);
    await replaySession(GAME, "example-1", JSON.parse(JSON.stringify(recording)));
});
