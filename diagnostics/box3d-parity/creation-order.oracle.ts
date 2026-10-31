import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createApp, hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { rainGround, rainSpawner } from "./creation-order-rain";
import { run } from "./native";
import { sampleBinary } from "./sample";

setDefaultTimeout(0);

test("Rain's direct and public authoring order equals native and rejects reversed joints", async () => {
    const steps = 600;
    const native = run([sampleBinary(), "rain", String(steps)]).trim().split("\n");
    const hashes = (text: string) => text.trim().split("\n").map(line => line.split("0x").at(-1)!);
    const directory = mkdtempSync(join(tmpdir(), "rain-creation-order-"));
    try {
        const original = readFileSync(join(import.meta.dir, "scenes.ts"), "utf8")
            .replace(/(from\s+|import\()("|')(\.\.?\/[^"']+)\2/g,
                (_match, prefix, _quote, path) => `${prefix}${JSON.stringify(resolve(import.meta.dir, path))}`);
        const reference = join(directory, "reference.ts");
        writeFileSync(reference, original);
        expect(hashes(run([process.execPath, reference, "rain", "1", String(steps)]))).toEqual(native);
        const reversed = original
            .replace("{ createRevoluteJoint,", "{ createRevoluteJoint as immediateRevoluteJoint,")
            .replace("{ createSphericalJoint,", "{ createSphericalJoint as immediateSphericalJoint,")
            .replace("{ createFilterJoint,", "{ createFilterJoint as immediateFilterJoint,")
            .replace("scene.step(state, i);", "scene.step(state, i); flushJoints();");
        // One step only: these mutation-only handles cannot be used for destruction.
        const wrappers = `
const pendingJoints: (() => void)[] = [];
function flushJoints() { for (let i = pendingJoints.length - 1; i >= 0; i--) pendingJoints[i](); pendingJoints.length = 0; }
function createRevoluteJoint(...args: Parameters<typeof immediateRevoluteJoint>) {
    pendingJoints.push(() => immediateRevoluteJoint(...args)); return { joint: -1 };
}
function createSphericalJoint(...args: Parameters<typeof immediateSphericalJoint>) {
    pendingJoints.push(() => immediateSphericalJoint(...args)); return { joint: -1 };
}
function createFilterJoint(...args: Parameters<typeof immediateFilterJoint>) {
    pendingJoints.push(() => immediateFilterJoint(...args)); return { joint: -1 };
}
`;
        const path = join(directory, "reversed.ts");
        writeFileSync(path, reversed.replace("const [name, threadArg, stepArg]", wrappers + "\nconst [name, threadArg, stepArg]"));
        const actual = hashes(run([process.execPath, path, "rain", "1", "1"]));
        expect(actual).toHaveLength(1);
        expect(actual[0], "reversed joint creation must fail the native step-0 hash").not.toBe(native[0]);
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        try {
            rainGround(app.world);
            const spawn = rainSpawner();
            for (let step = 0; step < steps; step++) {
                spawn(app.world, step);
                app.world.step(1 / 60);
                expect(hashPhysics(app.world).toString(16).padStart(16, "0"), `public Rain step ${step}`).toBe(native[step]);
            }
        } finally { await app.dispose(); }
        console.log(`Rain: ${steps} native, direct and public hashes equal; reversed joints differ at step 0 (${actual[0]} versus ${native[0]})`);
    } finally { rmSync(directory, { recursive: true, force: true }); }
});
