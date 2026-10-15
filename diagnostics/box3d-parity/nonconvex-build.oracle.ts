import { expect, test } from "bun:test";
import gold from "../../crates/physics/tests/nonconvex_build.gold.json";
import { init, kernel } from "../../src/standard/physics/kernel/kernel";
import { nativeBinary, run } from "./native";

for (const heightField of [false, true]) {
    test(`${heightField ? "height-field" : "mesh"} construction and generators equal native byte images`, async () => {
        const native = JSON.parse(run([nativeBinary("nonconvex-build.c")])) as typeof gold;
        expect(native).toEqual(gold);
        await init(undefined, { threads: 0 });
        const k = kernel(undefined);
        let cases = 0, images = 0;
        for (const c of native) {
            if ((c.operation >= 10) !== heightField) continue;
            const vertices = c.vertices ?? [], indices = c.indices ?? [], heights = c.heights ?? [], materials = c.materials ?? [];
            const ptr = k.nonconvexBuildInputPtr(vertices.length + indices.length + heights.length + materials.length);
            const input = new Uint32Array(k.memory.buffer, ptr, 16 + vertices.length + indices.length + heights.length + materials.length);
            input.set(c.args);
            if (c.operation === 0) {
                const flags = new Float32Array(new Uint32Array(c.args).buffer);
                input[1] = Number(!!flags[1]); input[2] = Number(!!flags[2]); input[3] = Number(!!flags[3]); input[7] = Number(!!flags[4]);
                input[4] = vertices.length / 3; input[5] = indices.length; input[6] = Number(materials.length > 0);
                input.set(vertices, 16); input.set(indices, 16 + vertices.length); input.set(materials, 16 + vertices.length + indices.length);
            } else if (c.operation === 10) {
                input[8] = Number(materials.length > 0);
                input.set(heights, 16); input.set(materials, 16 + heights.length);
            }
            const output = k.nonconvexBuild(c.operation);
            if (c.words === null) expect(output, c.name).toBe(0);
            else {
                expect(output, c.name).not.toBe(0);
                expect(Array.from(new Uint32Array(k.memory.buffer, output, c.words.length)), c.name).toEqual(c.words);
                ++images;
            }
            ++cases;
        }
        console.log(JSON.stringify({ kind: heightField ? "height-field" : "mesh", cases, images }));
    });
}
