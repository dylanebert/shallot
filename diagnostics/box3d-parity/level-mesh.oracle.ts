import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import gold from "../../examples/physics/tests/level-mesh.gold.json";
import { drawMeshData } from "../../src/standard/physics/shapes/readgeometry";
import { levelMeshBinary } from "./level-mesh";

for (const name of [...Object.keys(gold), "voxel_mesh_03", "voxel_mesh_04"]) {
    test(`${name} encoded arrays equal native mesh loader`, () => {
        const binary = levelMeshBinary();
        const path = join(process.env.BOX3D ?? "", "data/meshes", `${name}.obj`);
        const input = Bun.spawnSync([binary, path, "input"]);
        expect(input.exitCode).toBe(0);
        const bytes = input.stdout;
        const verticesEnd = 12 + 12 * bytes.readInt32LE(0);
        const indicesEnd = verticesEnd + 4 * bytes.readInt32LE(4);
        const encoded = JSON.parse(readFileSync(join(import.meta.dir, `../../examples/physics/src/${name}.json`), "utf8"));
        expect(encoded).toEqual({
            vertices: bytes.subarray(12, verticesEnd).toString("base64"),
            indices: bytes.subarray(verticesEnd, indicesEnd).toString("base64"),
            materialIndices: bytes.subarray(indicesEnd).toString("base64"),
        });
        if (!Object.hasOwn(gold, name)) return;
        const result = Bun.spawnSync([binary, path, "build"]);
        expect(result.exitCode).toBe(0);
        const out = result.stdout;
        const mesh = drawMeshData(new DataView(out.buffer, out.byteOffset, out.byteLength), 0);
        expect(createHash("sha256").update(JSON.stringify(mesh)).digest("hex")).toBe(gold[name as keyof typeof gold]);
    });
}
