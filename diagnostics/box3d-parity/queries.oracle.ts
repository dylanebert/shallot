import { expect, setDefaultTimeout, test } from "bun:test";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { type HullData } from "@dylanebert/shallot/standard/physics";
import { buildHulls, castHeightField, heightField, hull, QuerySample, QuerySamplePlugin } from "../../examples/physics/src/queries";
import { run } from "./native";
import { sampleBinary } from "./sample";
setDefaultTimeout(0);
const bits = new DataView(new ArrayBuffer(4));
function hex(value: number): string { bits.setFloat32(0, value, true); return bits.getUint32(0, true).toString(16).padStart(8, "0"); }
function hullLine(h: HullData): string { return `hull ${[h.surfaceArea, ...h.points.flatMap(p => [p.x, p.y, p.z])].map(hex).join(" ")}`; }
for (const scene of ["heightField", "hull"] as const) {
    test(`${scene}: public release queries and builds equal every native bit`, async () => {
        const start = performance.now();
        const expected = (await Bun.file(new URL(`../../examples/physics/tests/${scene}.queries.gold.txt`, import.meta.url)).text()).trim().split("\n");
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin, QuerySamplePlugin] });
        try {
            const world = app.world;
            const actual: string[] = [];
            if (scene === "hull") {
                hull(world);
                const sample = world.resource(QuerySample);
                actual.push(hullLine(sample.hull!), hullLine(sample.clone!));
                buildHulls(world, h => actual.push(hullLine(h)));
                expect(actual).toHaveLength(4002);
            } else {
                heightField(world); world.tick();
                for (const radius of [Math.fround(0.1), 0]) {
                    world.resource(QuerySample).radius = radius;
                    castHeightField(world, (h, hit) => actual.push(`cast ${Number(hit)} ${[h.point.x, h.point.y, h.point.z, h.normal.x, h.normal.y, h.normal.z, h.fraction].map(hex).join(" ")}`));
                }
                expect(actual).toHaveLength(28556);
            }
            expect(actual.length).toBe(expected.length);
            for (let i = 0; i < expected.length; i++) expect(actual[i], `${scene} record ${i}`).toBe(expected[i]);
            console.log(`${scene}: ${actual.length} bit-exact records, ${(performance.now() - start).toFixed(0)} ms`);
        } finally { app.dispose(); }
    });
    test(`${scene}: native recording repeats and a sample source mutation changes it`, () => {
        const start = performance.now();
        const sequence = fileURLToPath(new URL("../../examples/physics/tests/heightField.controls.txt", import.meta.url));
        const args = [scene, scene === "hull" ? "1" : "2", ...(scene === "hull" ? [] : [sequence])];
        const native = run([sampleBinary(), ...args]);
        expect(run([sampleBinary(), ...args])).toBe(native);
        expect(native).toBe(readFileSync(new URL(`../../examples/physics/tests/${scene}.queries.gold.txt`, import.meta.url), "utf8"));
        const copy = mkdtempSync(join(tmpdir(), "box3d-query-mutation-"));
        try {
            mkdirSync(join(copy, "samples")); mkdirSync(join(copy, "shared"));
            for (const name of ["sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp"]) copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, "samples", name));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
            const path = join(copy, "samples/sample_benchmark.cpp");
            const source = readFileSync(path, "utf8");
            const before = scene === "hull" ? "g_randomSeed = 42;" : "m_radius = 0.1f;";
            expect(source.includes(before)).toBe(true);
            const start = source.indexOf(scene === "hull" ? "class BenchmarkHull :" : "class BenchmarkHeightField :");
            writeFileSync(path, source.slice(0, start) + source.slice(start).replace(before, scene === "hull" ? "g_randomSeed = 43;" : "m_radius = 0.2f;"));
            expect(run([sampleBinary(join(copy, "samples")), ...args])).not.toBe(native);
        } finally { rmSync(copy, { recursive: true, force: true }); }
        console.log(`${scene} native repeat/mutation: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
