import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Shape } from "../../core/physics";
import { RenderTransformHistory, readRenderGlobalTransform } from "../../core/rendering";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp, Time } from "../../engine";
import {
    ClientPrediction,
    IntentReceiver,
    IntentSender,
    PredictionMotion,
    PredictionPlugin,
    PredictionTarget,
    Replicate,
    ReplicationSender,
    seededLink,
} from ".";

setDefaultTimeout(TIMEOUT.node);
const schema = [Transform, GlobalTransform, Body, Shape, PredictionMotion];
async function scene() {
    const app = await createApp({ defaults: false, plugins: [PredictionPlugin] });
    const world = app.world;
    const body = world.create();
    world.add(body, Transform);
    world.add(body, Body, { type: BodyType.Dynamic, linearVelocity: [60, 0, 0], gravityScale: 0 });
    world.add(body, Shape);
    world.add(body, PredictionTarget);
    world.add(body, RenderTransformHistory);
    world.add(body, Replicate);
    world.tick();
    return { app, world, body };
}

test("late-input feedback paces a slow client; bursts replay once and drawn motion stays continuous at the prediction cap", async () => {
    const server = await scene();
    const client = await scene();
    const link = seededLink(812, { minDelay: 3, maxDelay: 5, loss: 0, duplication: 0.1 });
    let held = false;
    const connection = {
        close: link.b.close,
        send: link.b.send,
        receive: () => (held ? [] : link.b.receive()),
    };
    const sender = new ReplicationSender(server.world, schema, [new Map([[1, schema]])], 1, link.a);
    const input = new IntentSender(connection);
    const accepted = new IntentReceiver(link.a);
    let reconciliations = 0;
    const restore = client.world.restore.bind(client.world);
    client.world.restore = (snapshot) => {
        reconciliations++;
        restore(snapshot);
    };
    let fixed = 0;
    const prediction = new ClientPrediction(
        client.world,
        schema,
        connection,
        () => client.body,
        () => {},
        (_world, _outputs, confirmed) => {
            if (!confirmed) fixed++;
        },
        40,
    );
    prediction.receiver.entities.set(1, client.body);
    prediction.onMessages = (messages) => {
        input.flush(messages);
    };
    const pose = new Float32Array(12);
    let previous = 0;
    let holds = 0;
    let resumed = 0;
    let burstReplays = 0;
    let slowestBurstFrame = Infinity;
    let fastestBurstFrame = 0;
    let late = 0;
    let peakLead = prediction.lead;
    let minimumLead = Infinity;
    let maximumLead = -Infinity;
    const initialLead = prediction.lead;
    try {
        sender.record(1);
        sender.flush();
        for (let frame = 1; frame <= 3400; frame++) {
            link.advance(frame);
            held = (frame >= 1400 && frame < 1410) || (frame >= 1900 && frame < 2040);
            accepted.receive(sender.flush(), server.world.time.fixedTick + 1);
            if (frame === 1300) {
                expect(prediction.lead).toBeLessThan(peakLead);
                expect(Math.max(...accepted.inputs.keys())).toBeGreaterThanOrEqual(
                    server.world.time.fixedTick,
                );
            }
            const before = server.world.time.fixedTick;
            // The long outage is a stalled server, not a server simulating past the held client.
            // Far-behind adoption is checked separately over a 400-tick gap.
            server.world.step(frame >= 1900 && frame < 2040 ? 0 : Time.FIXED_DT / 2);
            if (server.world.time.fixedTick !== before) {
                sender.record(
                    server.world.time.fixedTick,
                    [...accepted.records.values()],
                    accepted.commandAge(server.world.time.fixedTick),
                );
                sender.flush();
            }
            reconciliations = 0;
            fixed = 0;
            const head = client.world.time.fixedTick;
            // This clock runs 20% slow. Tick dilation must recover both its drift and link delay.
            client.world.step(Time.FIXED_DT * 0.4);
            if (frame < 500) peakLead = Math.max(peakLead, prediction.lead);
            expect(reconciliations).toBeLessThanOrEqual(1);
            if (frame < 1900)
                expect(fixed).toBeLessThanOrEqual(prediction.maxAhead + Time.MAX_FIXED_STEPS);
            if (frame === 1410) burstReplays = reconciliations;
            const record = input.record(client.world.time.fixedTick + 1, "input", [1]);
            prediction.accept([record]);
            input.flush();
            readRenderGlobalTransform(client.world, client.body, pose);
            const x = pose[0]!;
            if (frame > 1300) {
                const advance = x - previous;
                expect(advance).toBeGreaterThanOrEqual(-0.001);
                expect(advance).toBeLessThan(0.76);
                // A short confirmation burst must not quantize the rendered body to tick boundaries.
                if (frame >= 1395 && frame <= 1415) {
                    expect(advance).toBeGreaterThan(0.3);
                    slowestBurstFrame = Math.min(slowestBurstFrame, advance);
                    fastestBurstFrame = Math.max(fastestBurstFrame, advance);
                }
                if (held && client.world.time.fixedTick === head) {
                    if (frame > 2020) {
                        expect(Math.abs(advance)).toBeLessThan(0.001);
                        holds++;
                    }
                }
                if (frame > 2040 && frame < 2070 && x > previous) resumed++;
            }
            previous = x;
            if (frame > 2700) {
                const lead = client.world.time.fixedTick - server.world.time.fixedTick;
                minimumLead = Math.min(minimumLead, lead);
                maximumLead = Math.max(maximumLead, lead);
            }
        }
        if (accepted.commandAge(server.world.time.fixedTick)! > 0) late++;
        expect(late).toBe(0);
        expect(prediction.lead).toBeGreaterThan(initialLead);
        // 40 ticks beyond the latest received confirmation; after convergence the actual
        // server lead is nonnegative and no larger than that same conservative cap.
        expect(minimumLead).toBeGreaterThanOrEqual(0);
        expect(maximumLead).toBeLessThanOrEqual(40);
        expect(burstReplays).toBe(1);
        expect(fastestBurstFrame - slowestBurstFrame).toBeLessThan(0.08);
        expect(holds).toBeGreaterThan(10);
        expect(resumed).toBeGreaterThan(10);
    } finally {
        client.app.dispose();
        server.app.dispose();
    }
});
