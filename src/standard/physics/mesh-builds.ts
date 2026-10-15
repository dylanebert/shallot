// Built meshes for a World's PhysicsMeshes. A build is kept per registry id with the definition it was
// built from, so every Shape naming that id shares one MeshData and one geometry-database entry. A
// definition replaced by name rebuilds when a Shape next asks for it.

import { type PhysicsMesh, PhysicsMeshes } from "../../core/physics";
import type { Resource, World } from "../../engine";
import { createMesh, type MeshData } from "./shapes/mesh";

interface MeshBuild {
    def: PhysicsMesh;
    built: MeshData;
}

const MeshBuilds: Resource<(MeshBuild | undefined)[]> = { create: () => [] };

function build(world: World, def: PhysicsMesh): MeshData {
    const stride = def.stride || 3;
    const { vertices, indices, materialIndices } = def;
    const label = `physics: mesh "${def.name}"`;
    if (!Number.isInteger(stride) || stride < 3)
        throw new Error(`${label} has stride ${def.stride}, not 0 or an integer of at least 3`);
    if (vertices.length % stride !== 0 && vertices.length % stride !== 3)
        throw new Error(
            `${label} has ${vertices.length} vertex floats, not whole vertices at stride ${stride}`,
        );
    if (indices.length % 3 !== 0)
        throw new Error(`${label} has ${indices.length} indices, not a multiple of 3`);
    const vertexCount = Math.ceil(vertices.length / stride);
    for (let i = 0; i < indices.length; i++)
        if (indices[i] >= vertexCount)
            throw new Error(
                `${label} index ${i} is ${indices[i]}, past its ${vertexCount} vertices`,
            );
    if (materialIndices && materialIndices.length !== indices.length / 3)
        throw new Error(
            `${label} has ${materialIndices.length} material indices for ${indices.length / 3} triangles`,
        );
    const points = new Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) {
        const o = i * stride;
        points[i] = { x: vertices[o], y: vertices[o + 1], z: vertices[o + 2] };
    }
    const built = createMesh(
        {
            vertices: points,
            indices,
            materialIndices,
            weldTolerance: def.weldTolerance,
            weldVertices: def.weldVertices,
            useMedianSplit: def.useMedianSplit,
            identifyEdges: def.identifyEdges,
            clockwiseWinding: def.clockwiseWinding,
        },
        world,
    );
    if (!built) throw new Error(`${label} could not be built`);
    return built;
}

/** The built mesh for a PhysicsMeshes id, or undefined when none is registered; throws for a definition a solver refuses. */
export function meshBuild(world: World, id: number): MeshData | undefined {
    const meshes = world.resource(PhysicsMeshes);
    const name = meshes.name(id);
    const def = name === undefined ? undefined : meshes.get(name);
    const builds = world.resource(MeshBuilds);
    if (!def) {
        // A deleted mesh's input and build stay collectable instead of living until dispose.
        builds[id] = undefined;
        return undefined;
    }
    const slot = builds[id];
    if (slot?.def === def) return slot.built;
    const built = build(world, def);
    builds[id] = { def, built };
    return built;
}

/** Current builds by registry id, by reference; MeshData is immutable. Releases builds whose definition was deleted or replaced. */
export function captureMeshBuilds(world: World): [number, MeshData][] {
    const meshes = world.resource(PhysicsMeshes);
    const builds = world.resource(MeshBuilds);
    const captured: [number, MeshData][] = [];
    for (let id = 0; id < builds.length; id++) {
        const slot = builds[id];
        const name = meshes.name(id);
        if (!slot) continue;
        if (name !== undefined && meshes.get(name) === slot.def) captured.push([id, slot.built]);
        else builds[id] = undefined;
    }
    return captured;
}

/** Re-pairs captured builds with the definitions core physics restored, so restored Shapes keep their mesh identity. */
export function restoreMeshBuilds(world: World, captured: [number, MeshData][]): void {
    const meshes = world.resource(PhysicsMeshes);
    const builds = world.resource(MeshBuilds);
    builds.length = 0;
    for (const [id, built] of captured) {
        const name = meshes.name(id);
        const def = name === undefined ? undefined : meshes.get(name);
        if (def) builds[id] = { def, built };
    }
}
