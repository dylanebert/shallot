import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { PlayerMotion } from "../../../examples/first-person/src/motion";
import { Player, PlayerInput, PlayerPlugin } from "../../../examples/first-person/src/player";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { VisualCorrection } from "../../core/rendering";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp, type World } from "../../engine";
import {
    Character,
    hashPhysics,
    PhysicsEvents,
    setLinearVelocity,
    setTransform,
} from "../../standard/physics";
import {
    type AcceptedIntent,
    ClientPrediction,
    contactOutputKey,
    IntentReceiver,
    type IntentRecord,
    IntentSender,
    PredictionMotion,
    PredictionPlugin,
    PredictionTarget,
    Replicate,
    ReplicationSender,
    seededLink,
} from ".";

setDefaultTimeout(TIMEOUT.node);
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
async function scene() {
    const app = await createApp({
        defaults: false,
        plugins: [PlayerPlugin, PredictionPlugin],
        setup(world) {
            const camera = world.create();
            world.add(camera, Transform);
            const player = world.create();
            world.add(player, Transform, { translation: [2, 1.45, 15] });
            world.add(player, Body, { type: BodyType.Kinematic });
            world.add(player, Shape, {
                kind: ShapeKind.Capsule,
                capsuleCenter1: [0, -0.6, 0],
                capsuleCenter2: [0, 0.6, 0],
                radius: 0.3,
                enableContactEvents: 1,
            });
            world.add(player, Character);
            world.add(player, Player, { camera });
            const floor = world.create();
            world.add(floor, Transform, { translation: [0, 0, -5], scale: [32, 1, 52] });
            world.add(floor, Body, { type: BodyType.Static });
            world.add(floor, Shape, { enableContactEvents: 1 });
            const box = world.create();
            world.add(box, Transform, { translation: [2, 0.9, 12], scale: [0.8, 0.8, 0.8] });
            world.add(box, Body, { type: BodyType.Dynamic });
            world.add(box, PredictionTarget);
            world.add(box, Shape, { enableContactEvents: 1 });
            for (const eid of world.query([Transform])) world.add(eid, Replicate);
        },
    });
    app.world.tick();
    return app;
}
function drive(world: World, tick: number, records: readonly IntentRecord[]) {
    const player = [...world.query([Player])][0]!;
    if (!world.has(player, PlayerInput)) world.add(player, PlayerInput);
    const input = records.filter((r) => r.kind === "input").at(-1);
    if (input) world.storage(PlayerInput).move.set(player, input.values[0]!, input.values[1]!);
    world
        .storage(PlayerInput)
        .jumpPresses.set(
            player,
            records.filter((r) => r.kind === "operation" && r.tick <= tick).length,
        );
}
function poses(world: World) {
    return [...world.query([Body, GlobalTransform])].map(
        (eid) => world.get(eid, GlobalTransform)!.translation,
    );
}
function gameplay(world: World) {
    return [...world.query([Player])].map((eid) => [
        world.get(eid, Player),
        world.get(eid, PlayerInput),
        world.get(eid, PlayerMotion),
        world.get(eid, Character),
    ]);
}
async function run(switched = false, press = false, stepped = true) {
    const server = await scene(),
        client = await scene();
    const world = server.world,
        local = client.world;
    const link = seededLink(777, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
    const sender = new ReplicationSender(world, schema, [new Map([[1, schema]])], 1, link.a);
    let dropNext = false,
        dropped = 0;
    const sent = new IntentSender({
            receive: link.b.receive,
            send(bytes) {
                if (dropNext) {
                    dropNext = false;
                    dropped++;
                    return;
                }
                link.b.send(bytes);
            },
        }),
        accepted = new IntentReceiver(link.a);
    const outputs: string[] = [],
        observed: string[] = [];
    const player = [...local.query([Player])][0]!;
    const box = [...local.query([Body])].at(-1)!;
    let base = 1;
    const samples: {
        tick: number;
        base: number;
        confirmed: boolean;
        poses: ReturnType<typeof poses>;
        hash: bigint;
        gameplay: ReturnType<typeof gameplay>;
    }[] = [];
    const prediction = new ClientPrediction(
        local,
        schema,
        link.b,
        (world) => [...world.query([Player])][0],
        drive,
        (w, delivery, confirmed) => {
            samples.push({
                tick: w.time.fixedTick,
                base,
                confirmed,
                poses: poses(w),
                hash: hashPhysics(w),
                gameplay: gameplay(w),
            });
            const events = w.resource(PhysicsEvents);
            for (const kind of ["contactBegin", "contactEnd"] as const)
                for (const event of events[kind].records.slice(0, events[kind].count)) {
                    const key = contactOutputKey(kind, event);
                    observed.push(`${w.time.fixedTick}:${key}`);
                    delivery.deliver(w.time.fixedTick, key, () =>
                        outputs.push(`${w.time.fixedTick}:${key}`),
                    );
                }
        },
    );
    prediction.onMessages = (messages) => {
        sent.flush(messages);
    };
    // This composition already has the image's entities, in the same deterministic creation order.
    for (const [index, eid] of [...world.query([Replicate])].entries())
        prediction.receiver.entities.set(index + 1, eid);
    const archive = new Map<number, ReturnType<typeof poses>>();
    const hashes = new Map<number, bigint>();
    const states = new Map<number, ReturnType<typeof gameplay>>();
    let maximum = 0,
        settled = 0,
        pushed = 0,
        launch = 0;
    let enabled = true;
    const switches: number[] = [];
    const contacts: number[] = [];
    try {
        prediction.switchBody(box, true);
        sender.record(1);
        sender.flush();
        prediction.receiver.flush();
        sender.flush();
        function produce() {
            const tick = local.time.fixedTick + 1;
            const move =
                tick >= 30 && tick < 85
                    ? [Math.floor(tick / 9) % 2 ? 0.08 : -0.08, 1]
                    : tick >= 100 && tick < 170
                      ? [1, 0]
                      : tick >= 220 && tick < 290
                        ? [-1, 0]
                        : tick >= 300 && tick < 355
                          ? [0, 1]
                          : [0, 0];
            sent.record(tick, "input", move);
            if (press && tick === 43) {
                sent.record(tick, "operation", [1]);
                dropNext = true;
            }
            if (switched) {
                const a = local.get(player, GlobalTransform)!.translation;
                const b = local.get(box, GlobalTransform)!.translation;
                const inside = Math.hypot(...a.map((v, i) => v - b[i]!)) <= 3.5;
                if (inside !== enabled) {
                    enabled = inside;
                    switches.push(tick);
                    prediction.switchBody(box, inside);
                }
            }
            prediction.accept(sent.records.values());
            if (stepped) local.step(1 / 60);
            else local.tick();
        }
        while (local.time.fixedTick < 11) produce();
        for (let tick = 2; tick <= 400; tick++) {
            if (local.time.fixedTick < 400) produce();
            sent.flush();
            accepted.receive(sender.flush(), tick);
            drive(
                world,
                tick,
                [...accepted.records.values()]
                    .filter((r) => r.tick <= tick)
                    .sort((a, b) => a.tick - b.tick || a.id - b.id),
            );
            world.tick();
            archive.set(tick, poses(world));
            hashes.set(tick, hashPhysics(world));
            states.set(tick, gameplay(world));
            if (tick === 43) launch = world.storage(Character).velocity.y.get(player);
            const events = world.resource(PhysicsEvents);
            if (
                [
                    ...events.contactBegin.records.slice(0, events.contactBegin.count),
                    ...events.contactEnd.records.slice(0, events.contactEnd.count),
                ].some((e) => world.resolve(e.bodyA) === box || world.resolve(e.bodyB) === box)
            )
                contacts.push(tick);
            if (tick === 30) pushed = poses(world).at(-1)![2]!;
            if (tick === 90) pushed -= poses(world).at(-1)![2]!;
            sender.record(tick, [...accepted.records.values()]);
            sender.flush();
            base = tick;
            if (stepped) local.step(0);
            else sent.flush(prediction.receiver.flush());
        }
        let corrected = 0,
            hashDifferences = 0,
            replayed = 0,
            settledCount = 0;
        const changed = contacts.find((tick) => tick > (switches[1] ?? 400));
        for (const sample of samples) {
            if (sample.tick < 2) continue;
            const expected = archive.get(sample.tick)!;
            if (sample.hash !== hashes.get(sample.tick)) hashDifferences++;
            if (!switched) expect(sample.gameplay).toEqual(states.get(sample.tick)!);
            const error = Math.max(
                ...sample.poses.map((p, i) => Math.hypot(...p.map((v, j) => v - expected[i]![j]!))),
            );
            if (sample.confirmed) corrected = Math.max(corrected, error);
            else {
                maximum = Math.max(maximum, error);
                if (changed !== undefined && sample.base >= changed + 2) {
                    settled = Math.max(settled, error);
                    settledCount++;
                }
                replayed++;
            }
        }
        return {
            maximum,
            settled,
            corrected,
            hashDifferences,
            dropped,
            operations: [...accepted.records.values()].filter((r) => r.kind === "operation"),
            launch,
            replayed,
            pushed,
            outputs,
            observed,
            switches,
            contacts,
            changed,
            settledCount,
            confirmed: prediction.receiver.confirmed,
            confirmations: samples
                .filter((sample) => sample.confirmed)
                .map((sample) => sample.tick),
        };
    } finally {
        server.dispose();
        client.dispose();
    }
}
async function connectedPrediction(maxAhead = 32, confirmed: (world: World) => void = () => {}) {
    const server = await scene(),
        client = await scene();
    const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
    const sender = new ReplicationSender(server.world, schema, [new Map([[1, schema]])], 1, link.a);
    const prediction = new ClientPrediction(
        client.world,
        schema,
        link.b,
        (world) => [...world.query([Player])][0],
        drive,
        (world, _outputs, isConfirmed) => {
            if (isConfirmed) confirmed(world);
        },
        maxAhead,
    );
    for (const [index, eid] of [...client.world.query([Replicate])].entries())
        prediction.receiver.entities.set(index + 1, eid);
    sender.record(1);
    sender.flush();
    client.world.step(0);
    return { server, client, sender, prediction };
}

for (const lag of [1, 40]) {
    test(`a client ${lag} server ticks behind confirmations catches up exactly`, async () => {
        const hashes = new Map<number, bigint>();
        const pair = await connectedPrediction(32, (world) => {
            if (world.time.fixedTick > 1)
                expect(hashPhysics(world)).toBe(hashes.get(world.time.fixedTick)!);
        });
        const { server, client, sender } = pair;
        const records: AcceptedIntent[] = [];
        const origin = client.world.snapshot();
        try {
            for (let tick = 2; tick <= lag + 1; tick++) {
                records.push({ id: tick, tick, kind: "input", values: [0.1, 1], lateness: 0 });
                drive(server.world, tick, records);
                server.world.tick();
                hashes.set(tick, hashPhysics(server.world));
                sender.record(tick, records);
            }
            expect(client.world.time.fixedTick).toBe(1);
            sender.flush();
            client.world.step(0);
            expect(client.world.time.fixedTick).toBe(lag + 1);
            expect(hashPhysics(client.world)).toBe(hashPhysics(server.world));
            expect(gameplay(client.world)).toEqual(gameplay(server.world));
            // An older local image puts one ordered confirmation many ticks ahead without a wire gap.
            client.world.restore(origin);
            const next = lag + 2;
            records.push({ id: next, tick: next, kind: "input", values: [0.1, 1], lateness: 0 });
            drive(server.world, next, records);
            server.world.tick();
            hashes.set(next, hashPhysics(server.world));
            sender.record(next, records);
            sender.flush();
            client.world.step(0);
            expect(client.world.time.fixedTick).toBe(next);
            expect(hashPhysics(client.world)).toBe(hashPhysics(server.world));
            expect(gameplay(client.world)).toEqual(gameplay(server.world));
        } finally {
            server.dispose();
            client.dispose();
        }
    });
}

test("a confirmation stall holds the predicted clock while frames draw, then resumes exactly without tick debt", async () => {
    const { server, client, sender, prediction } = await connectedPrediction(4);
    const local = client.world;
    const records: AcceptedIntent[] = Array.from({ length: 5 }, (_, index) => ({
        id: index + 2,
        tick: index + 2,
        kind: "input",
        values: [0.1, 1],
        lateness: 0,
    }));
    let draws = 0,
        frames = 0;
    local.addSystem({
        group: "draw",
        update() {
            draws++;
        },
    });
    local.addSystem({
        group: "simulation",
        update() {
            frames++;
        },
    });
    try {
        prediction.accept(records);
        local.step(4 / 60);
        expect(local.time.fixedTick).toBe(5);
        const held = hashPhysics(local);
        for (let i = 0; i < 40; i++) {
            local.step(4 / 60);
            expect(local.time.fixedTick).toBe(5);
            expect(local.time.fixedSteps).toBe(0);
            expect(hashPhysics(local)).toBe(held);
        }
        expect(draws).toBe(41);
        expect(frames).toBe(41);
        local.step(1 / 120);
        expect(local.time.fixedTick).toBe(5);
        expect(local.time.fixedAlpha).toBeCloseTo(0.5);
        local.step(1 / 120);
        expect(local.time.fixedTick).toBe(5);
        expect(local.time.fixedAlpha).toBeCloseTo(0);
        drive(
            server.world,
            2,
            records.filter((r) => r.tick <= 2),
        );
        server.world.tick();
        sender.record(2, records);
        sender.flush();
        for (let tick = 3; tick <= 6; tick++) {
            drive(
                server.world,
                tick,
                records.filter((r) => r.tick <= tick),
            );
            server.world.tick();
        }
        local.step(1 / 60);
        expect(local.time.fixedTick).toBe(6);
        expect(local.time.fixedSteps).toBe(1);
        expect(hashPhysics(local)).toBe(hashPhysics(server.world));
        expect(gameplay(local)).toEqual(gameplay(server.world));
        local.step(0);
        expect(local.time.fixedTick).toBe(6);
        expect(local.time.fixedSteps).toBe(0);
        expect(draws).toBe(45);
        expect(frames).toBe(45);
    } finally {
        server.dispose();
        client.dispose();
    }
});

test("confirmation error adds to the current decayed visual correction without entering fixed placement", async () => {
    const server = await scene(),
        client = await scene();
    const world = client.world;
    const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
    const sender = new ReplicationSender(server.world, schema, [new Map([[1, schema]])], 1, link.a);
    const avatar = [...world.query([Player])][0]!;
    const prediction = new ClientPrediction(
        world,
        schema,
        link.b,
        () => avatar,
        () => {},
    );
    try {
        for (const [index, eid] of [...world.query([Replicate])].entries())
            prediction.receiver.entities.set(index + 1, eid);
        sender.record(1);
        sender.flush();
        world.step(0);
        for (let i = 0; i < 4; i++) world.step(1 / 60);
        world.add(avatar, VisualCorrection, { translation: [0.25, 0, 0] });
        for (const tick of [2, 3]) {
            const old = world.storage(GlobalTransform).translation.x.get(avatar);
            const residual = world.storage(VisualCorrection).translation.x.get(avatar) * 0.5;
            world.storage(VisualCorrection).translation.x.set(avatar, residual);
            setTransform(server.world, avatar, [2 + tick * 0.2, 1.45, 15], [0, 0, 0, 1]);
            server.world.tick();
            sender.record(tick);
            sender.flush();
            world.step(0);
            const fixed = world.storage(GlobalTransform).translation.x.get(avatar);
            expect(fixed).not.toBe(old);
            expect(world.storage(VisualCorrection).translation.x.get(avatar)).toBeCloseTo(
                residual + old - fixed,
                6,
            );
            const snapshot = world.snapshot();
            world.storage(VisualCorrection).translation.x.set(avatar, 0.125);
            world.restore(snapshot);
            expect(world.storage(VisualCorrection).translation.x.get(avatar)).toBe(0.125);
            expect(world.storage(GlobalTransform).translation.x.get(avatar)).toBe(fixed);
        }
    } finally {
        server.dispose();
        client.dispose();
    }
});

let predicted: ReturnType<typeof run> | undefined;
const exact = () => (predicted ??= run());
test("tick-addressed whole-World replay matches the authoritative avatar and opted-in pushable exactly", async () => {
    const r = await exact();
    expect(r.confirmed).toBe(400);
    expect(r.confirmations).toEqual(Array.from({ length: 400 }, (_, i) => i + 1));
    expect(r.pushed).toBeGreaterThan(1);
    expect(r.maximum).toBe(0);
    expect(r.corrected).toBe(0);
    expect(r.hashDifferences).toBe(0);
    expect(r.replayed).toBeGreaterThan(3000);
});
test("explicit world ticks run the same preparation and capture as paced ticks", async () => {
    const r = await run(false, false, false);
    expect(r.maximum).toBe(0);
    expect(r.hashDifferences).toBe(0);
});

test("a pending press replays through ClientPrediction and stays exact after its first message is dropped", async () => {
    const r = await run(false, true);
    expect(r.dropped).toBe(1);
    expect(r.operations).toHaveLength(1);
    expect(r.operations[0]!.tick).toBe(43);
    expect(r.launch).toBeGreaterThan(8);
    expect(r.maximum).toBe(0);
    expect(r.hashDifferences).toBe(0);
});
test("a real contact is delivered once across repeated fixed ticks", async () => {
    const r = await exact();
    expect(r.outputs.length).toBeGreaterThan(0);
    expect(r.observed.length).toBeGreaterThan(r.outputs.length);
    expect(r.outputs.length).toBe(new Set(r.outputs).size);
    expect([...r.outputs].sort()).toEqual([...new Set(r.observed)].sort());
});
test("prediction switching stays within 4.6 mm and reconverges", async () => {
    const r = await run(true);
    expect(r.switches).toHaveLength(2);
    expect(r.changed).toBeDefined();
    expect(r.settledCount).toBeGreaterThan(0);
    expect(r.maximum).toBeGreaterThan(0);
    expect(r.maximum).toBeLessThanOrEqual(0.0046);
    expect(r.settled).toBe(0);
});

test("an image spawns live body motion, while a non-predicted body displays only confirmed placement", async () => {
    const server = await scene();
    const client = await createApp({ defaults: false, plugins: [PlayerPlugin, PredictionPlugin] });
    try {
        for (let i = 0; i < 7; i++) client.world.create();
        const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
        const serverBox = [...server.world.query([Body])].at(-1)!;
        setLinearVelocity(server.world, serverBox, { x: 2, y: 0, z: 0 });
        const sender = new ReplicationSender(
            server.world,
            schema,
            [new Map([[1, schema]])],
            1,
            link.a,
        );
        const prediction = new ClientPrediction(
            client.world,
            schema,
            link.b,
            (world) => [...world.query([Player])][0],
            () => {},
        );
        sender.record(1);
        sender.flush();
        prediction.receiver.flush();
        const box = prediction.receiver.entities.get(4)!;
        expect(client.world.storage(Body).linearVelocity.x.get(box)).toBe(2);
        const confirmed = client.world.get(box, GlobalTransform)!.translation;
        for (let i = 0; i < 3; i++) client.world.tick();
        expect(client.world.get(box, GlobalTransform)!.translation).toEqual(confirmed);
        expect(prediction.predicted(box)).toBe(false);
        prediction.switchBody(box, true);
        client.world.tick();
        expect(client.world.storage(GlobalTransform).translation.x.get(box)).toBeGreaterThan(
            confirmed[0],
        );
    } finally {
        server.dispose();
        client.dispose();
    }
});
