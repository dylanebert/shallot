import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, GlobalTransform, Transform } from "../../core";
import { component, type Plugin, u32 } from "../../engine";
import {
    ClientPrediction,
    type Game,
    IntentSender,
    PredictionMotion,
    PredictionPlugin,
    Replicate,
    replaySession,
    type SessionRecording,
} from ".";
import type { RecordingCall } from "./recording";
import { createSession } from "./testing";

setDefaultTimeout(TIMEOUT.node);

const Count = component("RecordedPredictedCount", { value: u32 });
const Runtime = {
    create: (): {
        prediction?: ClientPrediction;
        input?: IntentSender;
        drop?: boolean;
        issued?: boolean;
        dropped: number;
    } => ({ dropped: 0 }),
};
const schema = [Count, Transform, GlobalTransform, Body, PredictionMotion];
const plugin: Plugin = {
    name: "RecordedPredictionGame",
    dependencies: [PredictionPlugin],
    components: [Count],
    initialize(world) {
        const eid = world.create();
        world.add(eid, Count);
        world.add(eid, Transform);
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Replicate);
    },
    systems: [
        {
            group: "setup",
            update(world) {
                const runtime = world.resource(Runtime);
                if (!runtime.prediction || runtime.issued || world.time.fixedTick < 24) return;
                runtime.issued = true;
                const operation = runtime.input!.record(
                    world.time.fixedTick + 1,
                    "operation",
                    [100],
                );
                runtime.prediction.accept([operation]);
                runtime.drop = true;
                runtime.input!.flush();
                runtime.prediction.switchBody([...world.query([Count])][0]!, true);
            },
        },
    ],
};
const game: Game = {
    plugins: [plugin],
    schema,
    maxClients: 2,
    rules: (_world, eid) => [new Map([[eid, schema]])],
    validateIntent: () => true,
    applyIntent(world, _eid, tick, operations, input) {
        const eid = [...world.query([Count])][0]!;
        const value = world.get(eid, Count)!.value;
        world
            .storage(Count)
            .value.set(
                eid,
                value +
                    (input?.values[0] ?? 0) +
                    operations.filter((operation) => operation.tick === tick).length * 100,
            );
    },
    onClient(world, connection) {
        const runtime = world.resource(Runtime);
        const routed = {
            get closed() {
                return connection.closed;
            },
            close: () => connection.close(),
            receive: () => connection.receive(),
            send(bytes: Uint8Array) {
                if (
                    runtime.drop &&
                    new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(
                        0,
                        true,
                    ) === 4
                ) {
                    runtime.drop = false;
                    runtime.dropped++;
                    return;
                }
                connection.send(bytes);
            },
        };
        const input = new IntentSender(routed);
        const prediction = new ClientPrediction(
            world,
            schema,
            routed,
            () => undefined,
            (world, tick, records, sample) => game.applyIntent(world, 0, tick, records, sample),
        );
        prediction.receiver.entities.set(1, [...world.query([Count])][0]!);
        prediction.sampleInput = (tick) => {
            const sample = input.record(tick, "input", [1]);
            input.flush();
            return sample;
        };
        prediction.onMessages = (messages) => input.flush(messages);
        runtime.input = input;
        runtime.prediction = prediction;
    },
};
function flatten(items: RecordingCall[]): RecordingCall[] {
    return items.flatMap((call) => [call, ...flatten(call.calls)]);
}

async function capture(subject = game) {
    const recordings = new Map<string, SessionRecording>();
    const session = await createSession(subject, {
        clients: 1,
        thinClients: 1,
        seed: 42,
        faults: { minDelay: 100, maxDelay: 100, loss: 0.2, duplication: 0 },
        recording: (role) => ({
            gameVersion: "prediction-1",
            write: (recording) => {
                recordings.set(role, recording);
            },
        }),
    });
    try {
        if (subject === game) {
            await session.advance(75);
            expect(session.clients[0]!.world.resource(Runtime).dropped).toBe(1);
        } else {
            await session.advance(25);
            const world = session.clients[0]!.world;
            const prediction = world.resource(Runtime).prediction!;
            prediction.accept([
                { id: 1, tick: world.time.fixedTick + 1, kind: "operation", values: [100] },
            ]);
            prediction.switchBody([...world.query([Count])][0]!, true);
            await session.advance(50);
        }
    } finally {
        await session.dispose();
    }
    return recordings;
}

test("prediction replays samples, speculative operations with their first message dropped, and body switches", async () => {
    const recordings = await capture();
    for (const recording of recordings.values())
        await replaySession(game, "prediction-1", JSON.parse(JSON.stringify(recording)));
});

test("an extra recorded prediction call or changed call order is reported at its entry tick", async () => {
    const original = (await capture()).get("client[0]")!;
    for (const reorder of [false, true]) {
        const recording = structuredClone(original);
        const parent = flatten(recording.calls).find((call) =>
            call.calls.some((child) => child.kind === "switchBody"),
        )!;
        const index = parent.calls.findIndex((call) => call.kind === "switchBody");
        const change = parent.calls[index]!;
        if (reorder) {
            parent.calls.splice(index, 1);
            parent.calls.unshift(change);
        } else parent.calls.splice(index, 0, structuredClone(change));
        await expect(replaySession(game, "prediction-1", recording)).rejects.toThrow(
            `fixed tick ${change.tick}`,
        );
    }
});

test("prediction calls made outside World systems are driven by playback once", async () => {
    const external = { ...game, plugins: [{ ...plugin, systems: [] }] };
    const recordings = await capture(external);
    const recording = recordings.get("client[0]")!;
    expect(recording.calls.filter((call) => call.kind === "accept")).toHaveLength(1);
    expect(recording.calls.filter((call) => call.kind === "switchBody")).toHaveLength(1);
    await replaySession(external, "prediction-1", recording);
});

test("removing a recorded sample, local operation or body switch reds at its entry tick", async () => {
    const original = (await capture()).get("client[0]")!;
    for (const kind of ["accept", "switchBody", "sample"]) {
        const recording = structuredClone(original);
        const parent = flatten(recording.calls).find((call) =>
            call.calls.some(
                (child) =>
                    child.kind === kind &&
                    (kind !== "accept" || JSON.stringify(child.args).includes("operation")),
            ),
        )!;
        expect(parent).toBeDefined();
        const index = parent.calls.findIndex(
            (call) =>
                call.kind === kind &&
                (kind !== "accept" || JSON.stringify(call.args).includes("operation")),
        );
        const removed = parent.calls[index]!;
        if (kind === "sample") removed.result = null;
        else parent.calls.splice(index, 1);
        await expect(replaySession(game, "prediction-1", recording)).rejects.toThrow(
            `fixed tick ${removed.tick}`,
        );
    }
});
