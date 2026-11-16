import { expect, test } from "bun:test";
import { readRenderGlobalTransform, VisualCorrection } from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
import { createApp, Time, World } from "../../engine";
import {
    ClientInterpolation,
    InterpolationPlugin,
    InterpolationTarget,
    Replicate,
    ReplicationReceiver,
    ReplicationSender,
    seededLink,
} from ".";

for (const gap of [1, 3])
    test(`remote placement follows confirmed brackets with a ${gap}-tick confirmation gap`, async () => {
        const app = await createApp({ defaults: false, plugins: [InterpolationPlugin] });
        const world = app.world;
        const client = new ClientInterpolation(world);
        const eid = world.create();
        world.add(eid, GlobalTransform);
        world.add(eid, InterpolationTarget);
        client.confirm(0, [eid]);
        await world.step((gap === 3 ? gap + 2 : gap) * Time.FIXED_DT);
        world.storage(GlobalTransform).translation.set(eid, gap * 2, gap * 4, 0);
        world.storage(GlobalTransform).rotation.set(eid, 0, 0, 1, 0);
        world.storage(GlobalTransform).scale.set(eid, 2, 2, 2);
        client.confirm(gap, [eid]);
        if (gap === 3) expect(client.delay).toBeGreaterThan(Time.FIXED_DT * 1.25);
        // A late confirmation increases the arrival-jitter margin.
        await world.step(Time.FIXED_DT * 0.25);
        const target = gap + 0.25 - client.delay / Time.FIXED_DT;
        const alpha = Math.max(0, Math.min(1, target / gap));
        const out = new Float32Array(12);
        readRenderGlobalTransform(world, eid, out);
        expect(out[0]).toBeCloseTo(gap * 2 * alpha);
        expect(out[1]).toBeCloseTo(gap * 4 * alpha);
        const length = Math.hypot(alpha, 1 - alpha);
        expect(out[6]).toBeCloseTo(alpha / length);
        expect(out[7]).toBeCloseTo((1 - alpha) / length);
        expect(out[8]).toBe(2);
        expect(world.get(eid, GlobalTransform)!.translation).toEqual([gap * 2, gap * 4, 0]);
        // With the next confirmation dropped, hold the last bracket rather than extrapolate.
        for (let i = 0; i < 8; i++) await world.step(Time.FIXED_DT);
        readRenderGlobalTransform(world, eid, out);
        expect(out[0]).toBeCloseTo(gap * 2);
        expect(out[6]).toBeCloseTo(1);
        app.dispose();
    });

test("bounded confirmation history wraps without losing its delayed bracket or reusing a dead entity's path", async () => {
    const app = await createApp({ defaults: false, plugins: [InterpolationPlugin] });
    const world = app.world;
    const interpolation = new ClientInterpolation(world);
    const eid = world.create();
    world.add(eid, GlobalTransform);
    world.add(eid, InterpolationTarget);
    const out = new Float32Array(12);
    for (let tick = 0; tick < 512; tick++) {
        world.storage(GlobalTransform).translation.set(eid, tick * 2, 0, 0);
        interpolation.confirm(tick, [eid]);
        await world.step(Time.FIXED_DT * 0.25);
        readRenderGlobalTransform(world, eid, out);
        const target = Math.max(0, tick + 0.25 - interpolation.delay / Time.FIXED_DT);
        expect(out[0]).toBeCloseTo(target * 2, 3);
        await world.step(Time.FIXED_DT * 0.75);
    }
    world.destroy(eid);
    const replacement = world.create();
    expect(replacement).toBe(eid);
    world.add(replacement, GlobalTransform, { translation: [2000, 0, 0] });
    world.add(replacement, InterpolationTarget);
    interpolation.confirm(512, [replacement]);
    await world.step(Time.FIXED_DT * 0.25);
    readRenderGlobalTransform(world, replacement, out);
    expect(out[0]).toBe(2000);
    app.dispose();
});

test("a predicted or owned selection stops interpolation before the next confirmation", async () => {
    const app = await createApp({ defaults: false, plugins: [InterpolationPlugin] });
    const world = app.world;
    let excluded = false;
    const interpolation = new ClientInterpolation(world, () => excluded);
    const eid = world.create();
    world.add(eid, GlobalTransform);
    world.add(eid, InterpolationTarget);
    interpolation.confirm(0, [eid]);
    await world.step(Time.FIXED_DT);
    world.storage(GlobalTransform).translation.set(eid, 4, 0, 0);
    interpolation.confirm(1, [eid]);
    await world.step(Time.FIXED_DT * 0.25);
    const out = new Float32Array(12);
    readRenderGlobalTransform(world, eid, out);
    expect(out[0]).toBeCloseTo(1);
    excluded = true;
    await world.step(Time.FIXED_DT * 0.25);
    readRenderGlobalTransform(world, eid, out);
    expect(out[0]).toBe(4);
    expect(world.has(eid, VisualCorrection)).toBe(false);
    app.dispose();
});

test("lost and reordered confirmations retain the confirmed path without changing fixed placement", async () => {
    const app = await createApp({ defaults: false, plugins: [InterpolationPlugin] });
    const world = app.world;
    const server = new World();
    const entity = server.create();
    server.add(entity, GlobalTransform);
    server.add(entity, Replicate);
    const link = seededLink(71, { minDelay: 1, maxDelay: 4, loss: 0.2, duplication: 0.2 });
    const schema = [GlobalTransform];
    const sender = new ReplicationSender(server, schema, [new Map([[1, schema]])], 1, link.a);
    const interpolation = new ClientInterpolation(world);
    const confirmations: { tick: number; arrival: number }[] = [];
    const receiver = new ReplicationReceiver(world, schema, link.b, (tick) => {
        const eid = receiver.entities.get(1)!;
        if (!world.has(eid, InterpolationTarget)) world.add(eid, InterpolationTarget);
        interpolation.confirm(tick, [eid]);
        confirmations.push({ tick, arrival: world.time.elapsed });
    });
    const out = new Float32Array(12);
    let fractional = 0;
    for (let frame = 0; frame < 70; frame++) {
        link.advance(frame);
        if (frame < 40) {
            server.storage(GlobalTransform).translation.set(entity, frame, frame * 2, 0);
            sender.record(frame);
        }
        sender.flush(frame);
        receiver.flush();
        await world.step(Time.FIXED_DT);
        const latest = confirmations.at(-1);
        if (!latest) continue;
        const eid = receiver.entities.get(1)!;
        readRenderGlobalTransform(world, eid, out);
        const target =
            latest.tick +
            (world.time.elapsed - latest.arrival - interpolation.delay) / Time.FIXED_DT;
        expect(out[0]).toBeCloseTo(Math.max(0, Math.min(latest.tick, target)), 4);
        expect(out[1]).toBeCloseTo(out[0]! * 2, 4);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(latest.tick);
        if (out[0]! > 0 && out[0]! < latest.tick) fractional++;
    }
    expect(link.counts.dropped).toBeGreaterThan(0);
    expect(fractional).toBeGreaterThan(0);
    expect(receiver.confirmed).toBe(39);
    server.dispose();
    app.dispose();
});

test("prediction error decays to identity in draw while fixed sees the correction immediately", async () => {
    let fixed = -1;
    const app = await createApp({
        defaults: false,
        plugins: [
            InterpolationPlugin,
            {
                name: "ObserveCorrectedFixed",
                systems: [
                    {
                        name: "observe-corrected-fixed",
                        group: "fixed",
                        update(world) {
                            fixed = world.storage(GlobalTransform).translation.x.get(eid);
                        },
                    },
                ],
            },
        ],
    });
    const world = app.world;
    const eid = world.create();
    world.add(eid, GlobalTransform, { translation: [10, 0, 0] });
    world.add(eid, VisualCorrection, { translation: [-4, 0, 0], rotation: [0, 0, 1, 0] });
    const out = new Float32Array(12);
    let previous = 4;
    for (let i = 0; i < 50; i++) {
        await world.step(Time.FIXED_DT);
        expect(fixed).toBe(10);
        readRenderGlobalTransform(world, eid, out);
        const error = Math.abs(out[0]! - 10);
        expect(error).toBeLessThanOrEqual(previous);
        previous = error;
    }
    expect(out[0]).toBe(10);
    expect(out[6]).toBe(0);
    expect(out[7]).toBe(1);
    expect(world.has(eid, VisualCorrection)).toBe(false);
    app.dispose();
});
