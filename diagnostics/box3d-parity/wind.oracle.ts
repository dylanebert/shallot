import { expect, setDefaultTimeout, test } from "bun:test";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { run } from "./native";
import { nativeSseOutput } from "./native-evidence";
import { BodyType } from "../../src/standard/physics/common/types";
import { init, kernel } from "../../src/standard/physics/kernel/kernel";
import { readStateAngularVelocity } from "../../src/standard/physics/kernel/bodycolumns";
import { makeBoxHull } from "../../src/standard/physics/shapes/build-hull";
import { createSphereShape, createCapsuleShape, createHullShape } from "../../src/standard/physics/shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef, stepTestWorld } from "../../src/standard/physics/testing";
import { bodyGetLinearVelocity, getBodyState, bodySetAwake, bodyDisable, bodySetType } from "../../src/standard/physics/world/body";

setDefaultTimeout(0);
await init(undefined, { threads: 0 });
const bits = new Uint32Array(1);
const floats = new Float32Array(bits.buffer);
function hex(n: number): string { floats[0] = n; return bits[0]!.toString(16).padStart(8, "0"); }

function correctedNative(): string[] {
    const directory = mkdtempSync(join(tmpdir(), "box3d-wind-wake-"));
    try {
        const source = join(directory, "source");
        cpSync(process.env.BOX3D!, source, { recursive: true, filter: path => ![".git", "build", "build-release"].includes(path.split("/").at(-1)!) });
        const path = join(source, "src/shape.c");
        const original = readFileSync(path, "utf8");
        const start = original.indexOf("void b3Shape_ApplyWind(");
        const declaration = "\tb3BodySim* sim = b3GetBodySim( world, body );\n";
        const selected = original.slice(start);
        expect(selected.includes(declaration)).toBe(true);
        const corrected = selected.replace(declaration, "").replace("\tb3BodyState* state = b3GetBodyState( world, body );", declaration + "\tb3BodyState* state = b3GetBodyState( world, body );");
        writeFileSync(path, original.slice(0, start) + corrected);
        const build = join(directory, "build");
        run(["cmake", "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_OSX_ARCHITECTURES=x86_64", "-DBOX3D_BENCHMARKS=OFF", "-DBOX3D_SAMPLES=OFF", "-DBOX3D_UNIT_TESTS=OFF"]);
        run(["cmake", "--build", build, "-j", "8", "--target", "box3d"]);
        const binary = join(directory, "wind");
        run(["cc", "-arch", "x86_64", "-O2", "-ffp-contract=off", `-I${join(source, "include")}`, join(import.meta.dir, "wind.c"), join(build, "src/libbox3d.a"), "-lm", "-o", binary]);
        return run(["arch", "-x86_64", binary]).trim().split("\n");
    } finally { rmSync(directory, { recursive: true, force: true }); }
}

test("wind shape kinds, drag, lift, maximum speed and wake equal native velocity bits", () => {
    const expected = nativeSseOutput("wind.c", "").trim().split("\n");
    const corrected = correctedNative();
    // Only the sleeping/wake=true rows use the fix for shape.c's freed simulation pointer.
    expect(corrected.slice(0, 18)).toEqual(expected.slice(0, 18));
    expect(corrected.slice(21)).toEqual(expected.slice(21));
    expect(corrected.slice(18, 21)).not.toEqual(expected.slice(18, 21));
    expected.splice(18, 3, ...corrected.slice(18, 21));
    const actual: string[] = [];
    for (let j = 0; j < 9; j++) {
        const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
        try {
            const bodies = Array.from({ length: 3 }, (_, i) => {
                const body = createTestBody(world, { type: BodyType.Dynamic, position: { x: i * 10, y: 0, z: 0 }, linearVelocity: { x: 1, y: 2, z: -3 }, angularVelocity: { x: 0.2, y: -0.3, z: 0.4 } });
                const def = testShapeDef();
                const shape = i === 0 ? createSphereShape(world, body, def, { center: { x: 0.2, y: 0, z: 0 }, radius: 0.5 }) : i === 1 ? createCapsuleShape(world, body, def, { center1: { x: 0, y: -0.5, z: 0 }, center2: { x: 0, y: 0.5, z: 0 }, radius: 0.25 }) : createHullShape(world, body, def, makeBoxHull(0.5, 0.75, 0.25));
                if (j === 5 || j === 6) bodySetAwake(world, body, false);
                if (j === 7) bodyDisable(world, body);
                if (j === 8) bodySetType(world, body, BodyType.Kinematic);
                kernel(world.ecsState).shapeApplyWind(world.worldId, shape!, 12, -4, 7, j === 1 ? 0 : 1, j === 2 ? 0 : 1, j === 3 ? 2 : 20, j === 4 || j === 6 ? 1 : 0);
                return body;
            });
            stepTestWorld(world, 1 / 60);
            for (const body of bodies) {
                const v = bodyGetLinearVelocity(world, body, { x: 0, y: 0, z: 0 });
                const state = getBodyState(world, body);
                const w = state === null ? { x: 0, y: 0, z: 0 } : readStateAngularVelocity(world, state, { x: 0, y: 0, z: 0 });
                actual.push([v.x, v.y, v.z, w.x, w.y, w.z].map(hex).join(" "));
            }
        } finally { destroyTestWorld(world); }
    }
    expect(actual).toEqual(expected);
});
