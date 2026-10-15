// Built geometry for a World's geometry registries, keyed weakly by the definition each was built
// from. Every Shape naming a definition shares one build and one geometry-database entry; a
// definition replaced by name is a new key, and a deleted one the game no longer holds is collected
// with its build.

import {
    type HeightField,
    HeightFields,
    type PhysicsMesh,
    PhysicsMeshes,
} from "../../core/physics";
import type { Registry, Resource, World } from "../../engine";
import { createHeightField, type HeightFieldData } from "./shapes/heightfield";
import { createMesh, type MeshData } from "./shapes/mesh";

interface Geometry<Def extends { name: string }, Built extends object> {
    registry: Resource<Registry<Def>>;
    builds: Resource<WeakMap<Def, Built>>;
    build(world: World, def: Def): Built;
}

function lookup<Def extends { name: string }, Built extends object>(
    world: World,
    geometry: Geometry<Def, Built>,
    id: number,
): Built | undefined {
    const registry = world.resource(geometry.registry);
    const name = registry.name(id);
    const def = name === undefined ? undefined : registry.get(name);
    if (!def) return undefined;
    const builds = world.resource(geometry.builds);
    let built = builds.get(def);
    if (!built) {
        built = geometry.build(world, def);
        builds.set(def, built);
    }
    return built;
}

function capture<Def extends { name: string }, Built extends object>(
    world: World,
    geometry: Geometry<Def, Built>,
): [number, Built][] {
    const registry = world.resource(geometry.registry);
    const builds = world.resource(geometry.builds);
    const captured: [number, Built][] = [];
    for (const def of registry) {
        const built = builds.get(def);
        if (built) captured.push([registry.id(def.name)!, built]);
    }
    return captured;
}

function restore<Def extends { name: string }, Built extends object>(
    world: World,
    geometry: Geometry<Def, Built>,
    captured: [number, Built][],
): void {
    const registry = world.resource(geometry.registry);
    const builds = world.resource(geometry.builds);
    for (const [id, built] of captured) {
        const name = registry.name(id);
        const def = name === undefined ? undefined : registry.get(name);
        if (def) builds.set(def, built);
    }
}

function buildMesh(world: World, def: PhysicsMesh): MeshData {
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

const meshes: Geometry<PhysicsMesh, MeshData> = {
    registry: PhysicsMeshes,
    builds: { create: () => new WeakMap() },
    build: buildMesh,
};

/** The built mesh for a PhysicsMeshes id, or undefined when none is registered; throws for a definition a solver refuses. */
export function meshBuild(world: World, id: number): MeshData | undefined {
    return lookup(world, meshes, id);
}

function buildHeightField(world: World, def: HeightField): HeightFieldData {
    const { heights, materialIndices, scale, countX, countZ } = def;
    const label = `physics: height field "${def.name}"`;
    if (!Number.isInteger(countX) || !Number.isInteger(countZ) || countX < 2 || countZ < 2)
        throw new Error(`${label} is ${countX} by ${countZ} points, not integers of at least 2`);
    if (heights.length !== countX * countZ)
        throw new Error(`${label} has ${heights.length} heights for ${countX * countZ} points`);
    const cells = (countX - 1) * (countZ - 1);
    if (materialIndices && materialIndices.length !== cells)
        throw new Error(
            `${label} has ${materialIndices.length} material indices for ${cells} cells`,
        );
    if (!(scale[0] > 0 && scale[1] > 0 && scale[2] > 0))
        throw new Error(`${label} has scale [${scale.join(", ")}], not positive`);
    if (!(def.globalMinimumHeight <= def.globalMaximumHeight))
        throw new Error(
            `${label} has globalMinimumHeight ${def.globalMinimumHeight} above globalMaximumHeight ${def.globalMaximumHeight}`,
        );
    return createHeightField(
        {
            heights,
            materialIndices: materialIndices ?? null,
            scale: { x: scale[0], y: scale[1], z: scale[2] },
            countX,
            countZ,
            globalMinimumHeight: def.globalMinimumHeight,
            globalMaximumHeight: def.globalMaximumHeight,
            clockwiseWinding: def.clockwiseWinding ?? false,
        },
        world,
    );
}

const heightFields: Geometry<HeightField, HeightFieldData> = {
    registry: HeightFields,
    builds: { create: () => new WeakMap() },
    build: buildHeightField,
};

/** The built height field for a HeightFields id, or undefined when none is registered; throws for a definition a solver refuses. */
export function heightFieldBuild(world: World, id: number): HeightFieldData | undefined {
    return lookup(world, heightFields, id);
}

/** Builds of the registered definitions by registry id, by reference; builds are immutable. */
export interface GeometryBuilds {
    meshes: [number, MeshData][];
    heightFields: [number, HeightFieldData][];
}

export function captureGeometryBuilds(world: World): GeometryBuilds {
    return { meshes: capture(world, meshes), heightFields: capture(world, heightFields) };
}

/** Pairs captured builds with the definitions core physics restored, so restored Shapes keep their geometry identity. */
export function restoreGeometryBuilds(world: World, captured: GeometryBuilds): void {
    restore(world, meshes, captured.meshes);
    restore(world, heightFields, captured.heightFields);
}
