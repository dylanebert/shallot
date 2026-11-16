import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { PlayerMotion } from "../../../examples/first-person/src/motion";
import { Player, PlayerInput, PlayerPlugin } from "../../../examples/first-person/src/player";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp } from "../../engine";
import { Character } from "../../standard/physics";
import {
    ClientPrediction,
    IntentReceiver,
    IntentSender,
    PredictionMotion,
    PredictionPlugin,
    Replicate,
    ReplicationSender,
    seededLink,
} from ".";
import { encodeIntent, IntentMessageKind } from "./intent";

setDefaultTimeout(TIMEOUT.node);
const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };
async function scene() {
    const app = await createApp({ defaults: false, plugins: [PlayerPlugin, PredictionPlugin] });
    const world = app.world;
    const avatars: number[] = [];
    for (const x of [-3, 3]) {
        const camera = world.create();
        world.add(camera, Transform);
        const eid = world.create();
        world.add(eid, Transform, { translation: [x, 2, 0] });
        world.add(eid, Body, { type: BodyType.Kinematic });
        world.add(eid, Shape, { kind: ShapeKind.Capsule });
        world.add(eid, Character);
        world.add(eid, Player, { camera });
        world.add(eid, PlayerInput);
        avatars.push(eid);
    }
    for (const eid of world.query([Transform])) world.add(eid, Replicate);
    world.tick();
    return { app, world, avatars };
}

test("two clients' connection-scoped IDs never drive the other's first-person avatar", async () => {
    const server = await scene();
    const clients = [await scene(), await scene()];
    const links = clients.map((_, i) => seededLink(i + 1, faults));
    const receivers = links.map((link) => new IntentReceiver(link.a));
    const senders = links.map((link) => new IntentSender(link.b));
    const schema = [
        Transform,
        GlobalTransform,
        Body,
        Shape,
        Character,
        Player,
        PlayerInput,
        PlayerMotion,
        PredictionMotion,
    ];
    const replication = links.map(
        (link, i) =>
            new ReplicationSender(
                server.world,
                schema,
                [new Map([[i + 1, schema]])],
                i + 1,
                link.a,
            ),
    );
    const predictions = clients.map(
        (client, i) =>
            new ClientPrediction(
                client.world,
                schema,
                links[i]!.b,
                () => client.avatars[i],
                (world, _tick, _records, record) => {
                    if (record)
                        world
                            .storage(PlayerInput)
                            .move.set(client.avatars[i]!, record.values[0]!, record.values[1]!);
                },
            ),
    );
    try {
        for (let i = 0; i < 2; i++) {
            for (const [index, eid] of [...clients[i]!.world.query([Replicate])].entries())
                predictions[i]!.receiver.entities.set(index + 1, eid);
            replication[i]!.record(1);
            replication[i]!.flush();
            clients[i]!.world.step(0);
        }
        for (let tick = 2; tick <= 50; tick++) {
            for (let i = 0; i < 2; i++) {
                const record = senders[i]!.record(tick, "input", [0, i === 0 ? 1 : 0]);
                predictions[i]!.accept([record]);
                senders[i]!.flush();
                receivers[i]!.receive(replication[i]!.flush(), tick);
                const input = receivers[i]!.inputAt(tick)!;
                server.world
                    .storage(PlayerInput)
                    .move.set(server.avatars[i]!, input.values[0]!, input.values[1]!);
                clients[i]!.world.step(1 / 60);
            }
            server.world.tick();
            for (let i = 0; i < 2; i++) {
                replication[i]!.record(tick, [...receivers[i]!.records.values()]);
                replication[i]!.flush();
                clients[i]!.world.step(0);
            }
            const remote = server.world.get(server.avatars[1]!, GlobalTransform)!.translation;
            expect(remote[0]).toBe(3);
            expect(remote[2]).toBe(0);
            const local = clients[1]!.world.get(
                clients[1]!.avatars[1]!,
                GlobalTransform,
            )!.translation;
            expect(local[0]).toBe(3);
            expect(local[2]).toBe(0);
        }
        expect(
            server.world.storage(GlobalTransform).translation.z.get(server.avatars[0]!),
        ).toBeLessThan(-1);
        expect(
            clients[0]!.world.storage(GlobalTransform).translation.z.get(clients[0]!.avatars[0]!),
        ).toBeLessThan(-1);
        expect(receivers[0]!.inputAt(50)!.values).not.toEqual(receivers[1]!.inputAt(50)!.values);
    } finally {
        server.app.dispose();
        for (const client of clients) client.app.dispose();
    }
});

test("prediction preparation follows the step accumulator and explicit ticks leave its fraction intact", async () => {
    const client = await scene();
    const world = client.world;
    const prepared: number[] = [];
    try {
        new ClientPrediction(
            world,
            [],
            seededLink(1, faults).b,
            () => client.avatars[0],
            (_world, tick) => {
                prepared.push(tick);
            },
        );
        world.step(1 / 120);
        expect(world.time.fixedTick).toBe(1);
        expect(world.time.fixedAlpha).toBeCloseTo(0.5);
        expect(prepared).toEqual([]);
        world.tick();
        expect(prepared).toEqual([2]);
        expect(world.time.fixedAlpha).toBeCloseTo(0.5);
        const scaledHalfTick = world.time.scale / 2;
        world.step(1 / 120);
        expect(prepared).toEqual([2, 3]);
        expect(world.time.fixedTick).toBe(3);
        expect(world.time.fixedSteps).toBe(1);
        expect(world.time.fixedAlpha).toBeCloseTo(0.5 + scaledHalfTick - 1);
    } finally {
        client.app.dispose();
    }
});

test("wire intent beyond the prediction bound is refused, including u32 maximum at server tick five", async () => {
    const server = await scene();
    const link = seededLink(1, faults);
    const receiver = new IntentReceiver(link.a, 10);
    try {
        link.b.send(
            encodeIntent(IntentMessageKind.Records, [
                { id: 1, tick: 15, kind: "operation", values: [1] },
                { id: 2, tick: 16, kind: "operation", values: [2] },
                { id: 3, tick: 0xffffffff, kind: "operation", values: [3] },
            ]),
        );
        receiver.receive(link.a.receive(), 5);
        expect([...receiver.records.keys()]).toEqual([1]);
        expect(receiver.records.get(1)!.tick).toBe(15);
    } finally {
        server.app.dispose();
    }
});
