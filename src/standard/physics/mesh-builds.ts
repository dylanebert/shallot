// Built meshes for a World's PhysicsMeshes, keyed weakly by the definition each was built from. Every
// Shape naming a definition shares one MeshData and one geometry-database entry; a definition replaced
// by name is a new key, and a deleted one the game no longer holds is collected with its build.

import { type PhysicsMesh, PhysicsMeshes } from "../../core/physics";
import type { Resource, World } from "../../engine";
import { createMesh, type MeshData } from "./shapes/mesh";

const MeshBuilds: Resource<WeakMap<PhysicsMesh, MeshData>> = { create: () => new WeakMap() };

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
    if (!def) return undefined;
    const builds = world.resource(MeshBuilds);
    let built = builds.get(def);
    if (!built) {
        built = build(world, def);
        builds.set(def, built);
    }
    return built;
}

/** Builds of the registered definitions by registry id, by reference; MeshData is immutable. */
export function captureMeshBuilds(world: World): [number, MeshData][] {
    const meshes = world.resource(PhysicsMeshes);
    const builds = world.resource(MeshBuilds);
    const captured: [number, MeshData][] = [];
    for (const def of meshes) {
        const built = builds.get(def);
        if (built) captured.push([meshes.id(def.name)!, built]);
    }
    return captured;
}

/** Pairs captured builds with the definitions core physics restored, so restored Shapes keep their mesh identity. */
export function restoreMeshBuilds(world: World, captured: [number, MeshData][]): void {
    const meshes = world.resource(PhysicsMeshes);
    const builds = world.resource(MeshBuilds);
    for (const [id, built] of captured) {
        const name = meshes.name(id);
        const def = name === undefined ? undefined : meshes.get(name);
        if (def) builds.set(def, built);
    }
}
