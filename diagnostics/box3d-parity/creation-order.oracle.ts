import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createApp, hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { rainGround, rainSpawner } from "./creation-order-rain";
import { run } from "./native";
import { sampleBinary } from "./sample";

setDefaultTimeout(0);

test("Rain's public authoring order equals native at every step; deferred joints distinguish the order", async () => {
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
        const deferred = original
            .replace("{ createRevoluteJoint,", "{ createRevoluteJoint as immediateRevoluteJoint,")
            .replace("{ createSphericalJoint,", "{ createSphericalJoint as immediateSphericalJoint,")
            .replace("{ createFilterJoint,", "{ createFilterJoint as immediateFilterJoint,")
            .replace("scene.step(state, i);", "scene.step(state, i); flushJoints();");
        const wrappers = `
const pendingJoints: (() => void)[] = [];
function flushJoints() { for (const create of pendingJoints) create(); pendingJoints.length = 0; }
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
        const path = join(directory, "deferred.ts");
        writeFileSync(path, deferred.replace("const [name, threadArg, stepArg]", wrappers + "\nconst [name, threadArg, stepArg]"));
        const actual = hashes(run([process.execPath, path, "rain", "1", "262"]));
        expect(actual.slice(0, 261)).toEqual(native.slice(0, 261));
        expect(native[261]).toBe("7d6a463f6abc1bed");
        expect(actual[261]).toBe("824b177370f62dc5");
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
        console.log(`Rain: ${steps} native, direct and public hashes equal; deferred control differs at step 261`);
    } finally { rmSync(directory, { recursive: true, force: true }); }
});
