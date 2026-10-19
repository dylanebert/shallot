import { expect, test } from "bun:test";
import { World } from "../../src/engine";
import { StandardPhysicsPlugin } from "../../src/standard/physics";
import { cloneAndTransformHull } from "../../src/standard/physics/shapes/build-hull";
import { hullImage } from "../../src/standard/physics/shapes/hullbytes";
import { drawHull } from "../../src/standard/physics/shapes/readgeometry";
import { nativeBinary, run } from "./native";

test("cloned hull images equal native through placements, nonuniform scales and reflections", async () => {
    const cases = JSON.parse(run([nativeBinary("hull-transform.c")])) as { source: number[]; args: number[]; result: number[] }[];
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    try {
        for (const [index, item] of cases.entries()) {
            const source = drawHull(new DataView(new Uint32Array(item.source).buffer), 0);
            const before = hullImage(source).slice();
            const a = item.args;
            const result = cloneAndTransformHull(source, {
                p: { x: a[0]!, y: a[1]!, z: a[2]! },
                q: { v: { x: a[3]!, y: a[4]!, z: a[5]! }, s: a[6]! },
            }, { x: a[7]!, y: a[8]!, z: a[9]! }, world)!;
            expect(result, `case ${index}`).not.toBeNull();
            const image = hullImage(result);
            expect(Array.from(new Uint32Array(image.buffer, image.byteOffset, image.length / 4)), `case ${index}`).toEqual(item.result);
            expect(hullImage(source), `case ${index} source unchanged`).toEqual(before);
        }
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});
