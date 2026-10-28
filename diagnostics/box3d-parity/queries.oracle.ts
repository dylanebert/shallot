import { expect, setDefaultTimeout, test } from "bun:test";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { digestMismatch, querySteps, recordDigest } from "../../examples/physics/tests/digest";
import { queryRecords } from "./records";
import { run } from "./native";
import { sampleBinary } from "./sample";
setDefaultTimeout(0);
for (const scene of ["heightField", "hull"] as const) {
    test(`${scene}: public release query and build digests equal every native step`, async () => {
        const start = performance.now();
        const expected = (await Bun.file(new URL(`../../examples/physics/tests/${scene}.queries.gold.txt`, import.meta.url)).text()).trim().split("\n");
        const records = await queryRecords(scene);
        expect(records.trimEnd().split("\n")).toHaveLength(scene === "hull" ? 4002 : 28556);
        const actual = querySteps(scene, records).map(recordDigest);
        expect(actual.length).toBe(expected.length);
        for (let step = 0; step < expected.length; step++) expect(actual[step], digestMismatch(scene, step)).toBe(expected[step]);
        console.log(`${scene}: ${actual.length} step digests, ${(performance.now() - start).toFixed(0)} ms`);
    });
    test(`${scene}: native recording repeats and a sample source mutation changes it`, () => {
        const start = performance.now();
        const sequence = fileURLToPath(new URL("../../examples/physics/tests/heightField.controls.txt", import.meta.url));
        const args = [scene, scene === "hull" ? "1" : "2", ...(scene === "hull" ? [] : [sequence])];
        const native = run([sampleBinary(), ...args]);
        expect(run([sampleBinary(), ...args])).toBe(native);
        const gold = readFileSync(new URL(`../../examples/physics/tests/${scene}.queries.gold.txt`, import.meta.url), "utf8").trimEnd().split("\n");
        for (const [step, records] of querySteps(scene, native).entries()) expect(recordDigest(records), digestMismatch(scene, step)).toBe(gold[step]);
        const copy = mkdtempSync(join(tmpdir(), "box3d-query-mutation-"));
        try {
            mkdirSync(join(copy, "samples")); mkdirSync(join(copy, "shared"));
            for (const name of ["sample_geometry.cpp", "sample_ragdoll.cpp", "sample_mesh.cpp", "sample_collision.cpp", "sample_compound.cpp", "sample_continuous.cpp", "sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp", "sample_bodies.cpp", "sample_shapes.cpp", "sample_character.cpp", "mover.h", "mover.cpp"]) copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, "samples", name));
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
