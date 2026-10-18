import { cube, Meshes, type World } from "@dylanebert/shallot";
import { Materials } from "@dylanebert/shallot/standard/rendering";
import * as d from "typegpu/data";
import { DetailKind } from "./detail";

type Box = [number, number, number, number, number, number];

/** One part of a detail kind: its mesh, its material handle's fields, and the kind's part count. */
export const DetailPart = d.struct({
    mesh: d.u32,
    materialType: d.u32,
    material: d.u32,
    alphaMode: d.u32,
    alphaCutoff: d.f32,
    count: d.u32,
});
function boxes(name: string, parts: Box[]) {
    const vertices: number[] = [],
        indices: number[] = [];
    for (const [x, y, z, sx, sy, sz] of parts) {
        const box = cube([sx, sy, sz]);
        const base = vertices.length / 8;
        for (let i = 0; i < box.vertices.length; i += 8) {
            vertices.push(
                box.vertices[i] + x,
                box.vertices[i + 1] + y,
                box.vertices[i + 2] + z,
                ...box.vertices.slice(i + 3, i + 8),
            );
        }
        for (const index of box.indices) indices.push(base + index);
    }
    return { name, vertices: new Float32Array(vertices), indices: new Uint32Array(indices) };
}
export function detailParts(world: World) {
    const meshes = world.resource(Meshes);
    const handles = [
        boxes("grass", [
            [-0.18, 0.22, 0, 0.1, 0.44, 0.1],
            [0, 0.3, 0.1, 0.1, 0.6, 0.1],
            [0.18, 0.18, -0.1, 0.1, 0.36, 0.1],
        ]),
        boxes("stem", [[0, 0.3, 0, 0.1, 0.6, 0.1]]),
        boxes("amber head", [[0, 0.65, 0, 0.38, 0.22, 0.38]]),
        boxes("cyan head", [[0, 0.65, 0, 0.38, 0.22, 0.38]]),
        boxes("rock", [
            [0, 0.15, 0, 0.7, 0.3, 0.6],
            [0.12, 0.32, 0, 0.4, 0.2, 0.4],
        ]),
        boxes("trunk", [[0, 1, 0, 0.4, 2, 0.4]]),
        boxes("canopy", [
            [0, 2.25, 0, 2.8, 1.3, 2.8],
            [0, 3.05, 0, 1.8, 0.5, 1.8],
        ]),
    ].map((mesh) => meshes.add(mesh));
    const materials = [
        [0.09, 0.18, 0.045],
        [0.08, 0.16, 0.04],
        [0.72, 0.4, 0.05],
        [0.04, 0.43, 0.53],
        [0.15, 0.14, 0.125],
        [0.06, 0.057, 0.052],
        [0.075, 0.16, 0.04],
    ].map(([r, g, b]) =>
        world.resource(Materials).add({
            ...Materials.defaults!,
            baseColor: d.vec4f(r, g, b, 1),
            metallic: 0,
            perceptualRoughness: 1,
        }),
    );
    const kinds: number[][] = [];
    kinds[DetailKind.none] = [];
    kinds[DetailKind.grass] = [0];
    kinds[DetailKind.amber] = [1, 2];
    kinds[DetailKind.cyan] = [1, 3];
    kinds[DetailKind.rock] = [4];
    kinds[DetailKind.tree] = [5, 6];
    const rows = kinds.flatMap((parts) =>
        [0, 1].map((i) => {
            const part = parts[i] ?? 0;
            const material = materials[part];
            return DetailPart({
                mesh: handles[part],
                materialType: material.type,
                material: material.material,
                alphaMode: material.alphaMode,
                alphaCutoff: material.alphaCutoff,
                count: parts.length,
            });
        }),
    );
    return { meshes: handles, materials, rows };
}
