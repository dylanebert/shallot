import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp } from "../../engine";
import { ClientPrediction, PredictionPlugin, Replicate, ReplicationSender, seededLink } from ".";

setDefaultTimeout(TIMEOUT.node);
const schema = [Transform, GlobalTransform];
async function scene() {
    const app = await createApp({ defaults: false, plugins: [PredictionPlugin] });
    const w = app.world;
    const local = w.create();
    const remote = w.create();
    for (const eid of [local, remote]) {
        w.add(eid, Transform);
        w.add(eid, Replicate);
    }
    w.addSystem(
        {
            group: "fixed",
            update(w) {
                w.storage(Transform).translation.x.set(local, w.time.fixedTick);
                w.storage(Transform).translation.x.set(remote, w.time.fixedTick % 2 ? 0 : 10);
            },
        },
        "Prediction",
    );
    w.tick();
    return { app, w, local, remote };
}

test("a confirmation burst preserves accepted operations, outputs, callbacks and remote turns but restores and replays once", async () => {
    const server = await scene();
    const client = await scene();
    const link = seededLink(19, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
    const sender = new ReplicationSender(server.w, schema, [new Map([[1, schema]])], 1, link.a);
    const confirmations: number[] = [];
    const callbacks: number[] = [];
    const samples: number[][] = [];
    const outputs: number[] = [];
    let replayTicks = 0;
    const prediction = new ClientPrediction(
        client.w,
        schema,
        link.b,
        () => client.local,
        () => {},
        (w, delivery, confirmed) => {
            if (!confirmed) {
                replayTicks++;
                return;
            }
            confirmations.push(w.time.fixedTick);
            for (const record of prediction.records.values())
                if (record.kind === "operation" && record.tick === w.time.fixedTick)
                    delivery.deliver(record.tick, `${record.id}`, () => outputs.push(record.id));
        },
    );
    for (const eid of [client.local, client.remote]) prediction.receiver.entities.set(eid, eid);
    Object.defineProperty(prediction.receiver, "confirm", {
        value: (tick: number) => callbacks.push(tick),
    });
    const confirm = prediction.interpolation.confirm.bind(prediction.interpolation);
    prediction.interpolation.confirm = (tick, entities) => {
        samples.push([tick, client.w.storage(GlobalTransform).translation.x.get(client.remote)]);
        confirm(tick, entities);
    };
    let restores = 0;
    const restore = client.w.restore.bind(client.w);
    client.w.restore = (snapshot) => {
        restores++;
        restore(snapshot);
    };
    try {
        sender.record(1, [{ id: 1, tick: 1, kind: "operation", values: [9], lateness: 0 }]);
        for (let tick = 2; tick <= 3; tick++) {
            server.w.tick();
            sender.record(tick);
        }
        sender.flush();
        while (client.w.time.fixedTick < 8) client.w.tick();
        replayTicks = 0;
        client.w.step(0);
        expect([...prediction.records.keys()]).toEqual([1]);
        expect(confirmations).toEqual([1, 2, 3]);
        expect(callbacks).toEqual([1, 2, 3]);
        expect(samples).toEqual([
            [1, 0],
            [2, 10],
            [3, 0],
        ]);
        expect(outputs).toEqual([1]);
        expect(restores).toBe(1);
        expect(replayTicks).toBe(5);
        client.w.step(0);
        expect(outputs).toEqual([1]);
    } finally {
        client.app.dispose();
        server.app.dispose();
    }
});

test("late-input lead relaxes after a transient with hysteresis and ignores duplicate feedback", async () => {
    const client = await scene();
    const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
    const prediction = new ClientPrediction(
        client.w,
        schema,
        link.b,
        () => client.local,
        () => {},
    );
    try {
        prediction.accept([{ id: 1, tick: 22, kind: "input", values: [], lateness: 20 }]);
        const peak = prediction.lead;
        for (let id = 2; id <= 15; id++)
            prediction.accept([{ id, tick: id + 21, kind: "input", values: [], lateness: 0 }]);
        expect(prediction.lead).toBe(peak);
        for (let id = 16; id <= 99; id++)
            prediction.accept([{ id, tick: id + 21, kind: "input", values: [], lateness: 0 }]);
        expect(prediction.lead).toBeLessThan(peak);
        const settled = prediction.lead;
        prediction.accept(prediction.records.values());
        expect(prediction.lead).toBe(settled);
        for (let id = 100; id <= 197; id++)
            prediction.accept([
                { id, tick: 1000, kind: "input", values: [], lateness: 0, slack: 20 },
            ]);
        expect(prediction.lead).toBe(settled);
        for (let id = 198; id <= 295; id++)
            prediction.accept([
                { id, tick: id + 1000, kind: "input", values: [], lateness: 0, slack: 2 },
            ]);
        expect(prediction.lead).toBe(settled);
        for (let id = 296; id <= 393; id++)
            prediction.accept([
                { id, tick: id + 1000, kind: "input", values: [], lateness: 0, slack: 6 },
            ]);
        expect(prediction.lead).toBeLessThan(settled);
    } finally {
        client.app.dispose();
    }
});

test("an image hundreds of ticks ahead adopts its tick with at most lead-many fixed callbacks", async () => {
    const server = await scene();
    const client = await scene();
    const link = seededLink(3, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
    const sender = new ReplicationSender(server.w, schema, [new Map([[1, schema]])], 1, link.a);
    const prediction = new ClientPrediction(
        client.w,
        schema,
        link.b,
        () => client.local,
        () => {},
    );
    let fixed = 0;
    client.w.addSystem(
        {
            group: "fixed",
            update: () => {
                fixed++;
            },
        },
        "Prediction",
    );
    for (const eid of [client.local, client.remote]) prediction.receiver.entities.set(eid, eid);
    try {
        server.w.jumpToTick(400);
        server.w.tick();
        sender.record(401);
        sender.flush();
        client.w.step(0);
        expect(fixed).toBeLessThanOrEqual(Math.ceil(prediction.lead));
        expect(client.w.time.fixedTick).toBeGreaterThanOrEqual(401);
        expect(client.w.time.fixedTick).toBeLessThanOrEqual(401 + Math.ceil(prediction.lead));
        expect(client.w.storage(GlobalTransform).translation.x.get(client.local)).toBe(
            client.w.time.fixedTick,
        );
    } finally {
        client.app.dispose();
        server.app.dispose();
    }
});
