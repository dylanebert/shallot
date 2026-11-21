import { expect, test } from "bun:test";
import { component, createApp, entity, type Plugin, u32, type World } from "../../engine";
import {
    type Game,
    Replicate,
    ReplicationReceiver,
    replaySession,
    ServerSession,
    type SessionRecording,
} from ".";
import { type RecordingCall, SessionRecorder } from "./recording";
import { createSession } from "./testing";

const Count = component("RecordedCount", { value: u32 });
const Runtime = { create: (): { receiver?: ReplicationReceiver } => ({}) };
const plugin: Plugin = {
    name: "RecordedCounter",
    components: [Count],
    systems: [
        {
            group: "fixed",
            update(world) {
                world.resource(Runtime).receiver?.flush();
                for (const eid of world.query([Count]))
                    world.storage(Count).value.set(eid, world.get(eid, Count)!.value + 1);
            },
        },
    ],
};
const game: Game = {
    plugins: [plugin],
    schema: [Count],
    maxClients: 3,
    onCreate(world) {
        const eid = world.create();
        world.add(eid, Count);
        world.add(eid, Replicate);
    },
    rules: (_world, eid) => [new Map([[eid, [Count]]])],
    onJoin(world) {
        for (const eid of world.query([Count]))
            world.storage(Count).value.set(eid, world.get(eid, Count)!.value + 100);
    },
    onLeave(world) {
        for (const eid of world.query([Count]))
            world.storage(Count).value.set(eid, world.get(eid, Count)!.value + 1000);
    },
    validateIntent: () => true,
    applyIntent() {},
    onClient(world, connection) {
        world.resource(Runtime).receiver = new ReplicationReceiver(world, [Count], connection);
    },
};
test("recording preserves the live byte endpoint but playback never claims it as its source", () => {
    const options = { gameVersion: "counter-1", write() {} };
    const live = new SessionRecorder(game, "client", options);
    const transport = { endpoint: "wss://game.example/", close() {}, send() {}, receive: () => [] };
    expect(live.connection(transport).endpoint).toBe(transport.endpoint);
    const playback = new SessionRecorder(game, "client", options, live.recording);
    expect(playback.connection(transport).endpoint).toBeUndefined();
});

for (const role of ["server", "client"] as const) {
    test(`${role} hashes ignore presentation rows but detect replicated changes`, async () => {
        const recorder = new SessionRecorder(game, role, { gameVersion: "counter-1", write() {} });
        const app = await createApp({
            defaults: false,
            plugins: [recorder.plugin, { name: "HashMembership", components: [Count, Replicate] }],
        });
        try {
            const world = app.world;
            const replicated = world.create();
            world.add(replicated, Count, { value: 1 });
            if (role === "server") world.add(replicated, Replicate);
            else {
                const receiver = new ReplicationReceiver(world, [Count], {
                    send() {},
                    close() {},
                    receive: () => [],
                });
                receiver.entities.set(1, replicated);
            }
            world.tick();
            const presentation = world.create();
            world.add(presentation, Count, { value: 123 });
            world.tick();
            world.storage(Count).value.set(presentation, 456);
            world.tick();
            world.storage(Count).value.set(replicated, 2);
            world.tick();
            const values = hashes(recorder.recording).map((call) => call.result);
            expect(values[1]).toBe(values[0]);
            expect(values[2]).toBe(values[0]);
            expect(values[3]).not.toBe(values[0]);
        } finally {
            app.dispose();
        }
    });
}

function calls(recording: SessionRecording): RecordingCall[] {
    const flatten = (items: RecordingCall[]): RecordingCall[] =>
        items.flatMap((call) => [call, ...flatten(call.calls)]);
    return flatten(recording.calls);
}
function hashes(recording: SessionRecording) {
    return calls(recording).filter((call) => call.kind === "hash");
}
async function capture() {
    const recordings = new Map<string, SessionRecording>();
    const session = await createSession(game, {
        clients: 1,
        thinClients: 1,
        seed: 17,
        faults: { minDelay: 100, maxDelay: 100, loss: 0.2, duplication: 0 },
        recording: (role) => ({
            gameVersion: "counter-1",
            write: (recording) => {
                recordings.set(role, recording);
            },
        }),
    });
    try {
        await session.advance(80);
    } finally {
        await session.dispose();
    }
    return recordings;
}

test("a full and thin session at 100 ms and 20% loss replays both recorded roles from bytes", async () => {
    const recordings = await capture();
    expect([...recordings.keys()].sort()).toEqual(["client[0]", "server"]);
    for (const recording of recordings.values()) {
        expect(hashes(recording)).toHaveLength(80);
        await replaySession(game, "counter-1", JSON.parse(JSON.stringify(recording)));
    }
});

test("dropping one received result reports its first changed tick, not the last tick", async () => {
    const recording = (await capture()).get("client[0]")!;
    const received = calls(recording).find(
        (call) =>
            call.kind === "receive" &&
            (call.result as number[][]).some((bytes) => bytes.length > 16),
    )!;
    expect(received).toBeDefined();
    received.result = [];
    await expect(replaySession(game, "counter-1", recording)).rejects.toThrow(
        `fixed tick ${received.tick + 1}: hash result differs`,
    );
    expect(received.tick).toBeLessThan(80);
});

for (const field of ["kind", "tick", "args"] as const) {
    test(`call ${field} corruption is refused independently of its result and children`, async () => {
        const recording = (await capture()).get("server")!;
        const call = hashes(recording)[0]!;
        const tick = call.tick;
        if (field === "kind") call.kind = "not-a-hash";
        else if (field === "tick") call.tick++;
        else call.args = [123];
        await expect(replaySession(game, "counter-1", recording)).rejects.toThrow(
            `fixed tick ${tick}: expected`,
        );
    });
}

test("external pause and resume preserve the paused frame in replay", async () => {
    let recording: SessionRecording | undefined;
    const session = await createSession(game, {
        clients: 1,
        recording: (role) =>
            role === "client[0]"
                ? {
                      gameVersion: "counter-1",
                      write(value) {
                          recording = value;
                      },
                  }
                : undefined,
    });
    try {
        await session.advance(3);
        const world = session.clients[0]!.world;
        world.pause();
        world.step(1 / 60);
        expect(world.time.fixedTick).toBe(3);
        world.resume();
        world.step(1 / 60);
        expect(world.time.fixedTick).toBe(4);
    } finally {
        await session.dispose();
    }
    await replaySession(game, "counter-1", recording!);
});

test("external time scale and tick jumps drive the same step clock", async () => {
    let recording: SessionRecording | undefined;
    const session = await createSession(game, {
        clients: 0,
        recording: () => ({
            gameVersion: "counter-1",
            write(value) {
                recording = value;
            },
        }),
    });
    try {
        const world = session.server.world;
        world.step(0);
        world.setTimeScale(2);
        world.step(1 / 120);
        expect(world.time.fixedTick).toBe(1);
        world.setTimeScale(0);
        world.step(1 / 60);
        expect(world.time.fixedTick).toBe(1);
        world.setTimeScale(1);
        world.jumpToTick(20);
        world.step(1 / 60);
        expect(world.time.fixedTick).toBe(21);
    } finally {
        await session.dispose();
    }
    await replaySession(game, "counter-1", recording!);
    for (const kind of ["setTimeScale", "jumpToTick"]) {
        const changed = structuredClone(recording!);
        changed.calls.splice(
            changed.calls.findIndex((call) => call.kind === kind),
            1,
        );
        await expect(replaySession(game, "counter-1", changed)).rejects.toThrow(
            "Replay differs at fixed tick",
        );
    }
});

test("clock controls inside systems are checked in place without driving them twice", async () => {
    let updates = 0;
    const subject: Game = {
        ...game,
        plugins: [
            ...game.plugins,
            {
                name: "ClockControls",
                systems: [
                    {
                        group: "preFixed",
                        update(world) {
                            updates++;
                            world.pause();
                            world.resume();
                            world.setTimeScale(1);
                            world.discardOverstep(1 / 120);
                            world.jumpToTick(world.time.fixedTick + 1);
                        },
                    },
                ],
            },
        ],
    };
    let recording: SessionRecording | undefined;
    const session = await createSession(subject, {
        clients: 0,
        recording: () => ({
            gameVersion: "counter-1",
            write(value) {
                recording = value;
            },
        }),
    });
    try {
        for (let i = 0; i < 4; i++) session.server.world.step(1 / 60);
        expect(session.server.world.time.fixedTick).toBe(6);
    } finally {
        await session.dispose();
    }
    expect(updates).toBe(4);
    await replaySession(subject, "counter-1", recording!);
    expect(updates).toBe(8);
    for (const kind of ["pause", "resume", "setTimeScale", "discardOverstep", "jumpToTick"]) {
        const changed = structuredClone(recording!);
        const frame = changed.calls[0]!;
        const index = frame.calls.findIndex((call) => call.kind === kind);
        expect(index).toBeGreaterThanOrEqual(0);
        frame.calls.splice(index, 1);
        await expect(replaySession(subject, "counter-1", changed)).rejects.toThrow(
            "Replay differs at fixed tick 0: expected",
        );
    }
});

test("either engine or game version mismatch refuses replay", async () => {
    const recording = (await capture()).get("server")!;
    await expect(replaySession(game, "counter-2", recording)).rejects.toThrow("version mismatch");
    recording.engineVersion = "other-engine";
    await expect(replaySession(game, "counter-1", recording)).rejects.toThrow("version mismatch");
});

test("hashes detect different live gameplay, not just a recording read back as output", async () => {
    const recording = (await capture()).get("server")!;
    const changed: Game = {
        ...game,
        onCreate(world: World) {
            game.onCreate!(world);
            for (const eid of world.query([Count])) world.storage(Count).value.set(eid, 100);
        },
    };
    await expect(replaySession(changed, "counter-1", recording)).rejects.toThrow("fixed tick 1");
});

test("frame deltas, admission, explicit departure and transport closure are replay inputs", async () => {
    const recordings = new Map<string, SessionRecording>();
    const session = await createSession(game, {
        clients: 1,
        thinClients: 1,
        recording: (role) => ({
            gameVersion: "counter-1",
            write: (recording) => {
                recordings.set(role, recording);
            },
        }),
    });
    try {
        session.clients[0]!.world.step(0);
        session.clients[0]!.world.step(1 / 30);
        await session.advance(3);
        const authority = session.server.world.resource(ServerSession).server!;
        authority.disconnect([...authority.peers.keys()][1]!);
        session.clients[0]!.connection!.close();
        await session.advance(3);
    } finally {
        await session.dispose();
    }
    for (const recording of recordings.values()) await replaySession(game, "counter-1", recording);
    const server = recordings.get("server")!;
    for (const kind of ["connect", "disconnect"]) {
        const changed = structuredClone(server);
        const index = changed.calls.findIndex((call) => call.kind === kind);
        expect(index).toBeGreaterThanOrEqual(0);
        changed.calls.splice(index, 1);
        await expect(replaySession(game, "counter-1", changed)).rejects.toThrow(
            "Replay differs at fixed tick",
        );
    }
    const closed = structuredClone(server);
    const status = calls(closed).find((call) => call.kind === "closed" && call.result === true)!;
    expect(status).toBeDefined();
    status.result = false;
    await expect(replaySession(game, "counter-1", closed)).rejects.toThrow(
        `fixed tick ${status.tick}`,
    );
    const delta = structuredClone(recordings.get("client[0]")!);
    const frame = delta.calls.find((call) => call.kind === "step" && call.args[0] === 1 / 30)!;
    frame.args[0] = 0;
    await expect(replaySession(game, "counter-1", delta)).rejects.toThrow(
        `fixed tick ${frame.tick + 1}`,
    );
});

test("a client World's own entities shift no hashed row of entities its receiver created", async () => {
    const Link = component("RecordedLink", { target: entity });
    const linked: Game = {
        ...game,
        plugins: [plugin, { name: "RecordedLinks", components: [Link] }],
        schema: [Count, Link],
        onCreate(world) {
            const [a, b] = [world.create(), world.create()];
            for (const eid of [a, b]) world.add(eid, Replicate);
            world.add(a, Count);
            world.add(b, Link, { target: a });
        },
        rules: (_world, eid) => [new Map([[eid, [Count, Link]]])],
        onClient(world, connection) {
            world.resource(Runtime).receiver = new ReplicationReceiver(
                world,
                [Count, Link],
                connection,
            );
        },
    };
    let recording: SessionRecording | undefined;
    const session = await createSession(linked, {
        clients: 1,
        seed: 23,
        recording: (role) =>
            role === "server"
                ? undefined
                : { gameVersion: "counter-1", write: (value) => void (recording = value) },
    });
    try {
        // A tab creates its own entities, such as lights and text, that replay never composes.
        session.clients[0]!.world.create();
        await session.advance(20);
        const receiver = session.clients[0]!.world.resource(Runtime).receiver!;
        expect(receiver.entities.size).toBe(2);
    } finally {
        await session.dispose();
    }
    await replaySession(linked, "counter-1", JSON.parse(JSON.stringify(recording)));
});
