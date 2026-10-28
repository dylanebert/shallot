import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { createHash } from "node:crypto";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { init } from "../../../src/standard/physics/kernel/kernel";
import { createMesh } from "../../../src/standard/physics/shapes/build-nonconvex";
import building from "../src/building.json" with { type: "json" };
import conveyor from "../src/conveyor.json" with { type: "json" };
import { decodeMesh } from "../src/mesh-data";
import stairs from "../src/stairs.json" with { type: "json" };
import testMap from "../src/test_map01.json" with { type: "json" };
import voxel01 from "../src/voxel_mesh_01.json" with { type: "json" };
import voxel02 from "../src/voxel_mesh_02.json" with { type: "json" };
import gold from "./level-mesh.gold.json" with { type: "json" };

setDefaultTimeout(TIMEOUT.node);
beforeAll(() => init(undefined, { threads: 0 }));
const meshes = [
    ["conveyor", conveyor],
    ["test_map01", testMap],
    ["stairs", stairs],
    ["building", building],
    ["voxel_mesh_01", voxel01],
    ["voxel_mesh_02", voxel02],
] as const;
for (const [name, encoded] of meshes) {
    test(`${name} decoded mesh builds exactly as native CreateMeshData`, () => {
        const data = decodeMesh(encoded);
        const vertices = Array.from({ length: data.vertices.length / 3 }, (_, i) => ({
            x: data.vertices[3 * i]!,
            y: data.vertices[3 * i + 1]!,
            z: data.vertices[3 * i + 2]!,
        }));
        const definition = {
            vertices,
            indices: data.indices,
            materialIndices: data.materialIndices,
            useMedianSplit: name === "conveyor",
            identifyEdges: true,
            weldVertices: true,
            weldTolerance: 0.002,
        };
        if (name === "conveyor") {
            const roles = new Array<number>(48).fill(0);
            for (const [role, pair] of [
                [0, 4],
                [9, 12],
                [21, 38],
                [43, 46],
                [30, 33],
                [18, 24],
            ].entries())
                for (const triangle of pair) roles[triangle] = role + 1;
            expect(createMesh(definition)!.materialIndices).toEqual(roles);
            // ConveyorMesh changes materials after CreateMeshData; compare its unmodified build too.
            definition.materialIndices = Uint8Array.from(data.materialIndices, (_, i) => i % 3);
        }
        const mesh = createMesh(definition);
        expect(mesh).not.toBeNull();
        expect(createHash("sha256").update(JSON.stringify(mesh)).digest("hex")).toBe(
            gold[name as keyof typeof gold],
        );
    });
}
