import { expect, setDefaultTimeout, test } from "bun:test";
import type { EntityRef, World } from "@dylanebert/shallot";
import {
    createApp,
    GlobalTransform,
    hashPhysics,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot";
import { PhysicsEvents, setTargetTransform } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { contact, EventSample, EventSamplePlugin, hit, joint, motor, sensor } from "../src/events";
import { Grab, pick, release } from "../src/grab";
import { type Controls, changeControl, controlValues, scenes } from "../src/scenes";
import { sampleGold } from "./gold";

function eventRecords(world: World) {
    const sample = world.resource(EventSample);
    const events = world.resource(PhysicsEvents);
    const shape = (ref: EntityRef) => sample.shapeRefs.indexOf(ref) + 1;
    const body = (ref: EntityRef) => sample.bodyRefs.indexOf(ref) + 1;
    const pair = (e: {
        shapeA: EntityRef;
        bodyA: EntityRef;
        shapeB: EntityRef;
        bodyB: EntityRef;
    }) => [shape(e.shapeA), body(e.bodyA), shape(e.shapeB), body(e.bodyB)];
    const sensors = (records: typeof events.sensorBegin) =>
        records.records
            .slice(0, records.count)
            .map((e) => [
                shape(e.sensorShape),
                body(e.sensorBody),
                shape(e.visitorShape),
                body(e.visitorBody),
            ]);
    return {
        sensorBegin: sensors(events.sensorBegin),
        sensorEnd: sensors(events.sensorEnd),
        contactBegin: events.contactBegin.records.slice(0, events.contactBegin.count).map(pair),
        contactEnd: events.contactEnd.records.slice(0, events.contactEnd.count).map(pair),
        contactHit: events.contactHit.records
            .slice(0, events.contactHit.count)
            .map((e) => [
                ...pair(e),
                e.point.x,
                e.point.y,
                e.point.z,
                e.normal.x,
                e.normal.y,
                e.normal.z,
                e.approachSpeed,
                String(e.userMaterialIdA),
                String(e.userMaterialIdB),
            ]),
        joints: events.joints.records
            .slice(0, events.joints.count)
            .map((e) => [sample.jointRefs.indexOf(e.joint) + 1]),
    };
}

setDefaultTimeout(TIMEOUT.node);

for (const [name, construct] of [
    ["hit", hit],
    ["joint", joint],
    ["sensor", sensor],
    ["contact", contact],
    ["motor", motor],
] as const) {
    for (const controlled of name === "contact" || name === "motor" || name === "joint"
        ? [false, true]
        : [false]) {
        test(`${name} equals every native solve ${controlled ? "through controls and input" : "by default"}`, async () => {
            const start = performance.now();
            const gold = sampleGold(
                await Bun.file(
                    new URL(`./${name}${controlled ? ".controls" : ""}.gold.txt`, import.meta.url),
                ).text(),
            );
            const sequence = controlled
                ? (await Bun.file(new URL(`./${name}.controls.txt`, import.meta.url)).text())
                      .trim()
                      .split("\n")
                      .map((line) => {
                          const [step, key, value] = line.split("\t");
                          return { step: Number(step), key: key!, value: Number(value) };
                      })
                : [];
            const values = controlValues(name);
            let bodies: number[] = [];
            const used = new Set<string>();
            const app = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin, EventSamplePlugin],
                setup(world) {
                    bodies = construct(world);
                },
            });
            let cursor = 0;
            const trackDebris = name === "contact" && controlled;
            const occupiedSlots = new Set<number>();
            const coverage = { pickups: 0, breakoffs: 0, respawns: 0 };
            const broken = new Set<number>();
            const hitMaterials = new Set<string>();
            const eventCoverage = {
                sensorBegin: 0,
                sensorEnd: 0,
                contactBegin: 0,
                contactEnd: 0,
                contactHit: 0,
            };
            try {
                for (let step = 0; step < gold.length; step++) {
                    const sample = app.world.resource(EventSample);
                    const previousDebris = trackDebris ? sample.debris.slice() : [];
                    while (sequence[cursor]?.step === step) {
                        const change = sequence[cursor++]!;
                        if (change.key === "grabBody") {
                            const grab = app.world.resource(Grab);
                            const body = app.world.resolve(sample.bodyRefs[change.value - 1]!)!;
                            const p = app.world.storage(GlobalTransform).translation;
                            Object.assign(grab.origin, {
                                x: p.x.get(body),
                                y: p.y.get(body),
                                z: p.z.get(body) + 10,
                            });
                            Object.assign(grab.translation, { x: 0, y: 0, z: -20 });
                            pick(app.world);
                            expect(grab.body).toBe(body);
                        } else if (change.key === "grabY")
                            app.world.resource(Grab).point[1] = change.value;
                        else if (change.key === "release") release(app.world);
                        else if (change.key in app.world.resource(EventSample).keys) {
                            app.world.resource(EventSample).keys[
                                change.key as "W" | "S" | "A" | "D"
                            ] = change.value;
                        } else {
                            const controls: Controls | undefined = scenes[name].controls;
                            expect(controls?.[change.key]).toBeDefined();
                            changeControl(
                                app.world,
                                name,
                                bodies,
                                values,
                                change.key,
                                change.value,
                            );
                            used.add(change.key);
                        }
                    }
                    const grab = app.world.resource(Grab);
                    if (grab.mouse)
                        setTargetTransform(app.world, grab.mouse, grab.point, [0, 0, 0, 1]);
                    app.world.tick();
                    const records = app.world.resource(PhysicsEvents);
                    for (let i = 0; i < records.contactHit.count; i++)
                        hitMaterials.add(String(records.contactHit.records[i]!.userMaterialIdA));
                    for (const key of Object.keys(eventCoverage) as (keyof typeof eventCoverage)[])
                        eventCoverage[key] += records[key].count;
                    for (let i = 0; i < records.joints.count; i++)
                        broken.add(sample.jointRefs.indexOf(records.joints.records[i]!.joint) + 1);
                    if (trackDebris) {
                        for (let index = 0; index < sample.debris.length; index++) {
                            const eid = sample.debris[index]!;
                            const current = eid;
                            if (previousDebris[index] && previousDebris[index] !== current)
                                coverage.pickups++;
                            if (current && current !== previousDebris[index]) {
                                if (occupiedSlots.has(index)) coverage.respawns++;
                                occupiedSlots.add(index);
                            }
                        }
                        const events = app.world.resource(PhysicsEvents).contactBegin;
                        for (const event of events.records.slice(0, events.count)) {
                            const wall =
                                app.world.resolve(event.bodyA) === sample.walls
                                    ? event.bodyA
                                    : app.world.resolve(event.bodyB) === sample.walls
                                      ? event.bodyB
                                      : 0;
                            if (!wall) continue;
                            const playerShape = wall === event.bodyA ? event.shapeB : event.shapeA;
                            if (
                                playerShape !== app.world.ref(sample.core) &&
                                !app.world.resolve(playerShape)
                            )
                                coverage.breakoffs++;
                        }
                    }
                    // Native is sampled before its post-Step mutations; an incorrect mutation or mass
                    // recomputation changes the next solve's inputs and reds its following hash.
                    const expected = gold[step]!;
                    expect(hashPhysics(app.world), `step ${step} hash`).toBe(expected.hash);
                    expect(Object.values(eventRecords(app.world)), `step ${step} events`).toEqual(
                        expected.events,
                    );
                }
                if (name === "contact" && controlled) {
                    expect(coverage.pickups, "recorded input attaches debris").toBeGreaterThan(0);
                    expect(
                        coverage.breakoffs,
                        "recorded input scrapes attached debris off a wall",
                    ).toBeGreaterThan(0);
                    expect(
                        coverage.respawns,
                        "the sample reuses a vacated debris slot",
                    ).toBeGreaterThan(0);
                }
                expect(cursor).toBe(sequence.length);
                if (name === "joint" && controlled)
                    expect([...broken].sort()).toEqual([1, 2, 3, 4]);
                if (name === "sensor") {
                    expect(eventCoverage.sensorBegin).toBeGreaterThan(0);
                    expect(eventCoverage.sensorEnd).toBeGreaterThan(0);
                }
                if (name === "hit") {
                    expect(eventCoverage.contactHit).toBeGreaterThan(0);
                    expect([...hitMaterials].sort()).toEqual(["1", "2", "3", "4", "5", "6"]);
                }
                if (trackDebris) {
                    expect(eventCoverage.contactBegin).toBeGreaterThan(0);
                    expect(eventCoverage.contactEnd).toBeGreaterThan(0);
                }
                if (controlled)
                    expect([...used].sort()).toEqual(
                        Object.keys(scenes[name].controls ?? {}).sort(),
                    );
            } finally {
                app.dispose();
            }
            console.log(
                `${name}${controlled ? " controls" : " default"}: ${(performance.now() - start).toFixed(0)} ms; cheap budget 250 ms`,
            );
        });
    }
}
