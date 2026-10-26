import { createApp, hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { type HullData, PhysicsEvents } from "@dylanebert/shallot/standard/physics";
import { BenchmarkSamplePlugin } from "../../examples/physics/src/benchmarks";
import { buildHulls, castHeightField, heightField, hull, QuerySample, QuerySamplePlugin } from "../../examples/physics/src/queries";
import { createScene } from "../../examples/physics/src/scenes";
import { SensorSample } from "../../examples/physics/src/sensor";
import { eventRecord, querySteps } from "../../examples/physics/tests/digest";
import { sampleGold } from "../../examples/physics/tests/gold";
import { run } from "./native";
import { sampleBinary } from "./sample";
import { fileURLToPath } from "node:url";

export async function queryRecords(scene: "heightField" | "hull"): Promise<string> {
    const bits = new DataView(new ArrayBuffer(4));
    function hex(value: number): string { bits.setFloat32(0, value, true); return bits.getUint32(0, true).toString(16).padStart(8, "0"); }
    function hullLine(h: HullData): string { return `hull ${[h.surfaceArea, ...h.points.flatMap(p => [p.x, p.y, p.z])].map(hex).join(" ")}`; }
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin, QuerySamplePlugin] });
    try {
        const world = app.world;
        const actual: string[] = [];
        if (scene === "hull") {
            hull(world);
            const sample = world.resource(QuerySample);
            actual.push(hullLine(sample.hull!), hullLine(sample.clone!));
            buildHulls(world, h => actual.push(hullLine(h)));
        } else {
            heightField(world); world.tick();
            for (const radius of [Math.fround(0.1), 0]) {
                world.resource(QuerySample).radius = radius;
                castHeightField(world, (h, hit) => actual.push(`cast ${Number(hit)} ${[h.point.x, h.point.y, h.point.z, h.normal.x, h.normal.y, h.normal.z, h.fraction].map(hex).join(" ")}`));
            }
        }
        return `${actual.join("\n")}\n`;
    } finally { app.dispose(); }
}

export function sensorEvents(world: Parameters<typeof hashPhysics>[0]): (number | string)[][][] {
    const sample = world.resource(SensorSample);
    const events = world.resource(PhysicsEvents);
    const sensors = (channel: typeof events.sensorBegin) => channel.records.slice(0, channel.count).map(e => [
        sample.shapeRefs.indexOf(e.sensorShape) + 1, sample.bodyRefs.indexOf(e.sensorBody) + 1,
        sample.shapeRefs.indexOf(e.visitorShape) + 1, sample.bodyRefs.indexOf(e.visitorBody) + 1,
    ]);
    if (events.contactBegin.count + events.contactEnd.count + events.contactHit.count + events.joints.count !== 0) throw new Error("benchmarkSensor: unexpected non-sensor events");
    return [sensors(events.sensorBegin), sensors(events.sensorEnd), [], [], [], []];
}

if (import.meta.main) {
    const [scene, index] = process.argv.slice(2);
    const step = Number(index);
    if (!index || !Number.isInteger(step) || step < 0 || (scene !== "benchmarkSensor" && scene !== "heightField" && scene !== "hull") || step >= (scene === "benchmarkSensor" ? 3000 : scene === "heightField" ? 2 : 1)) throw new Error("usage: BOX3D=<checkout> bun diagnostics/box3d-parity/records.ts <benchmarkSensor|heightField|hull> <zero-based-step>");
    let native: string;
    let actual: string;
    if (scene === "benchmarkSensor") {
        const record = sampleGold(run([sampleBinary(), scene, String(step + 1)]))[step]!;
        native = eventRecord(record.hash, record.events);
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin, BenchmarkSamplePlugin], setup(world) { createScene(world, scene); } });
        try {
            for (let i = 0; i <= step; i++) app.world.tick();
            actual = eventRecord(hashPhysics(app.world), sensorEvents(app.world));
        } finally { app.dispose(); }
    } else {
        const sequence = fileURLToPath(new URL("../../examples/physics/tests/heightField.controls.txt", import.meta.url));
        native = querySteps(scene, run([sampleBinary(), scene, scene === "hull" ? "1" : "2", ...(scene === "hull" ? [] : [sequence])]))[step]!;
        actual = querySteps(scene, await queryRecords(scene))[step]!;
    }
    console.log(`native ${scene} step ${step}\n${native}Shallot ${scene} step ${step}\n${actual}`);
}
