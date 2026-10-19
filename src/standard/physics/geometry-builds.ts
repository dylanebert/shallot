// Built geometry for a World's geometry registries, keyed weakly by the definition each was built
// from. Shapes naming a definition at the same body scale share one build and geometry-database
// entry; each added definition is a new key, and a removed one the game no longer holds is collected
// with its build. A compound's build also records its children's builds, and is rebuilt once one
// of them is no longer current.

import {
    type Compound,
    Compounds,
    type HeightField,
    HeightFields,
    type Hull,
    Hulls,
    type PhysicsAssets,
    type PhysicsMesh,
    PhysicsMeshes,
} from "../../core/physics";
import type { Resource, World } from "../../engine";
import { MAX_HULL_VERTICES } from "./common/constants";
import type { Transform } from "./common/records";
import { defaultSurfaceMaterial } from "./common/types";
import {
    cloneAndTransformHull,
    createCylinder,
    createHull,
    makeBoxHull,
    makeTransformedBoxHull,
} from "./shapes/build-hull";
import { type CompoundData, createCompound, MAX_COMPOUND_MESH_MATERIALS } from "./shapes/compound";
import { createHeightField, type HeightFieldData } from "./shapes/heightfield";
import type { HullData } from "./shapes/hull";
import { createMesh, type MeshData } from "./shapes/mesh";

interface Geometry<Def extends { name: string }, Built extends object> {
    registry: Resource<PhysicsAssets<Def>>;
    builds: Resource<WeakMap<Def, Built>>;
    scaled: Resource<WeakMap<Def, Map<string, Built>>>;
    build(world: World, def: Def, sx?: number, sy?: number, sz?: number): Built;
    /** Whether a cached build still reflects the registries it was built from; default always. */
    current?(world: World, def: Def, built: Built, sx?: number, sy?: number, sz?: number): boolean;
}

function lookup<Def extends { name: string }, Built extends object>(
    world: World,
    geometry: Geometry<Def, Built>,
    id: number,
    sx = 1,
    sy = 1,
    sz = 1,
): Built | undefined {
    const registry = world.resource(geometry.registry);
    const def = registry.get(id);
    if (!def) return undefined;
    if (sx === 1 && sy === 1 && sz === 1) {
        const builds = world.resource(geometry.builds);
        let built = builds.get(def);
        if (!built || geometry.current?.(world, def, built) === false) {
            built = geometry.build(world, def);
            builds.set(def, built);
        }
        return built;
    }
    const scaled = world.resource(geometry.scaled);
    let builds = scaled.get(def);
    if (!builds) {
        builds = new Map();
        scaled.set(def, builds);
    }
    const key = `${sx},${sy},${sz}`;
    let built = builds.get(key);
    if (!built || geometry.current?.(world, def, built, sx, sy, sz) === false) {
        built = geometry.build(world, def, sx, sy, sz);
        builds.set(key, built);
    }
    return built;
}

function capture<Def extends { name: string }, Built extends object>(
    world: World,
    geometry: Geometry<Def, Built>,
): [number, Built, string?][] {
    const registry = world.resource(geometry.registry);
    const builds = world.resource(geometry.builds);
    const captured: [number, Built, string?][] = [];
    for (const [handle, def] of registry.entries()) {
        const built = builds.get(def);
        if (built) captured.push([handle, built]);
        const scaled = world.resource(geometry.scaled).get(def);
        if (scaled) for (const [key, built] of scaled) captured.push([handle, built, key]);
    }
    return captured;
}

function restore<Def extends { name: string }, Built extends object>(
    world: World,
    geometry: Geometry<Def, Built>,
    captured: [number, Built, string?][],
): void {
    const registry = world.resource(geometry.registry);
    const builds = world.resource(geometry.builds);
    for (const [id, built, key] of captured) {
        const def = registry.get(id);
        if (!def) continue;
        if (key === undefined) builds.set(def, built);
        else {
            const scaled = world.resource(geometry.scaled);
            let entries = scaled.get(def);
            if (!entries) {
                entries = new Map();
                scaled.set(def, entries);
            }
            entries.set(key, built);
        }
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
    scaled: { create: () => new WeakMap() },
    build: buildMesh,
};

/** The built mesh for a PhysicsMeshes id, or undefined when none is registered; throws for a definition a solver refuses. */
export function meshBuild(world: World, id: number): MeshData | undefined {
    return lookup(world, meshes, id);
}

function buildHeightField(world: World, def: HeightField, sx = 1, sy = 1, sz = 1): HeightFieldData {
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
            scale: { x: scale[0] * sx, y: scale[1] * sy, z: scale[2] * sz },
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
    scaled: { create: () => new WeakMap() },
    build: buildHeightField,
};

/** The built height field for a HeightFields id, or undefined when none is registered; throws for a definition a solver refuses. */
export function heightFieldBuild(
    world: World,
    id: number,
    sx = 1,
    sy = 1,
    sz = 1,
): HeightFieldData | undefined {
    return lookup(world, heightFields, id, sx, sy, sz);
}

/** Builds a box or point hull at per-axis scale; throws for a definition a solver refuses. */
export function buildHull(world: World, def: Hull, sx = 1, sy = 1, sz = 1): HullData {
    if (def.cylinder) {
        const { height, radius, yOffset, sides } = def.cylinder;
        if (sx === sz && sx > 0 && sy > 0)
            return createCylinder(height * sy, radius * sx, yOffset * sy, sides, world);
        return scaledHull(world, createCylinder(height, radius, yOffset, sides, world), sx, sy, sz);
    }
    if (def.halfExtents) {
        const [x, y, z] = def.halfExtents;
        if (![x, y, z].every((extent) => Number.isFinite(extent) && extent > 0))
            throw new Error(
                `physics: hull "${def.name}" requires positive finite box half-extents`,
            );
        if (def.transform) {
            const { position: p, rotation: q } = def.transform;
            const transform = {
                p: { x: p[0], y: p[1], z: p[2] },
                q: { v: { x: q[0], y: q[1], z: q[2] }, s: q[3] },
            };
            if (sx === 1 && sy === 1 && sz === 1)
                return makeTransformedBoxHull(x, y, z, transform, world);
            return scaledHull(world, makeTransformedBoxHull(x, y, z, transform, world), sx, sy, sz);
        }
        return makeBoxHull(x * Math.abs(sx), y * Math.abs(sy), z * Math.abs(sz), world);
    }
    const source = def.points;
    if (source.length % 3 !== 0)
        throw new Error(
            `physics: hull "${def.name}" has ${source.length} point coordinates, not a multiple of 3`,
        );
    const points = new Array(source.length / 3);
    for (let i = 0; i < points.length; i++)
        points[i] = { x: source[3 * i] * sx, y: source[3 * i + 1] * sy, z: source[3 * i + 2] * sz };
    const built = createHull(points, def.maxVertexCount ?? MAX_HULL_VERTICES, world);
    if (!built) throw new Error(`physics: hull "${def.name}" could not be built`);
    return built;
}

/** At identity placement, the asset builds at shape and body scale together. Otherwise the asset
 * builds at unit scale and clones once with Shape.scale and the placement, as
 * b3CreateTransformedHullShape does, then again with the body scale. */
export function buildPlacedHull(
    world: World,
    def: Hull,
    scale: readonly [number, number, number],
    position: readonly [number, number, number] = [0, 0, 0],
    rotation: readonly [number, number, number, number] = [0, 0, 0, 1],
    bodyScale: readonly [number, number, number] = [1, 1, 1],
): HullData {
    if (
        !position.every(Number.isFinite) ||
        !rotation.every(Number.isFinite) ||
        Math.abs(Math.hypot(...rotation) - 1) > 1e-5
    )
        throw new Error(
            "physics: hull placement requires a finite position and normalized rotation",
        );
    if (position.every((n) => n === 0) && rotation.every((n, i) => n === (i === 3 ? 1 : 0)))
        return buildHull(
            world,
            def,
            scale[0] * bodyScale[0],
            scale[1] * bodyScale[1],
            scale[2] * bodyScale[2],
        );
    let hull = cloneAndTransformHull(
        buildHull(world, def),
        {
            p: { x: position[0], y: position[1], z: position[2] },
            q: { v: { x: rotation[0], y: rotation[1], z: rotation[2] }, s: rotation[3] },
        },
        { x: scale[0], y: scale[1], z: scale[2] },
        world,
    );
    if (hull && bodyScale.some((n) => n !== 1))
        hull = cloneAndTransformHull(
            hull,
            { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            { x: bodyScale[0], y: bodyScale[1], z: bodyScale[2] },
            world,
        );
    if (!hull) throw new Error("physics: hull placement could not produce a valid hull");
    return hull;
}

// Nonuniform scale can turn a rotated box into a parallelepiped or a cylinder into an elliptical prism.
function scaledHull(world: World, hull: HullData, sx: number, sy: number, sz: number): HullData {
    const built = createHull(
        hull.points.map((p) => ({ x: p.x * sx, y: p.y * sy, z: p.z * sz })),
        hull.vertexCount,
        world,
    );
    if (!built) throw new Error("physics: scaled hull could not be built");
    return built;
}

// Compound children share hull builds at each body scale. A Shape of kind Hull builds its own.
const hulls: Geometry<Hull, HullData> = {
    registry: Hulls,
    builds: { create: () => new WeakMap() },
    scaled: { create: () => new WeakMap() },
    build: buildHull,
};

interface CompoundBuild {
    data: CompoundData;
    /** Each child's hull and mesh build, in child order, as the compound was built from them. */
    hulls: HullData[];
    meshes: MeshData[];
}

type Placement = NonNullable<Compound["hulls"]>[number]["transform"];

function placement({ position, rotation }: Placement, sx = 1, sy = 1, sz = 1): Transform {
    return {
        p: { x: position[0] * sx, y: position[1] * sy, z: position[2] * sz },
        q: { v: { x: rotation[0], y: rotation[1], z: rotation[2] }, s: rotation[3] },
    };
}

const vec3 = ([x, y, z]: [number, number, number], sx = 1, sy = 1, sz = 1) => ({
    x: x * sx,
    y: y * sy,
    z: z * sz,
});

function buildCompound(world: World, def: Compound, sx = 1, sy = 1, sz = 1): CompoundBuild {
    const label = `physics: compound "${def.name}"`;
    if (
        (def.spheres?.length || def.capsules?.length) &&
        (Math.abs(sx) !== Math.abs(sy) || Math.abs(sx) !== Math.abs(sz))
    )
        throw new Error(
            `${label} sphere or capsule child requires uniform body Transform.scale magnitudes`,
        );
    const hullChildren = def.hulls ?? [];
    const meshChildren = def.meshes ?? [];
    const hullBuilds = hullChildren.map((child, i) => {
        const built = lookup(world, hulls, child.hull, sx, sy, sz);
        if (!built) throw new Error(`${label} hull ${i} names missing hull ${child.hull}`);
        return built;
    });
    const meshBuilds = meshChildren.map((child, i) => {
        const built = lookup(world, meshes, child.mesh);
        if (!built) throw new Error(`${label} mesh ${i} names missing mesh ${child.mesh}`);
        const count = child.materials?.length ?? 1;
        if (count !== built.materialCount || count > MAX_COMPOUND_MESH_MATERIALS)
            throw new Error(
                `${label} mesh ${i} has ${count} materials for its mesh's ${built.materialCount}, at most ${MAX_COMPOUND_MESH_MATERIALS}`,
            );
        return built;
    });
    const data = createCompound({
        capsules: def.capsules?.map((child) => ({
            capsule: {
                center1: vec3(child.center1, sx, sy, sz),
                center2: vec3(child.center2, sx, sy, sz),
                radius: child.radius * Math.abs(sx),
            },
            material: child.material ?? defaultSurfaceMaterial(),
        })),
        hulls: hullChildren.map((child, i) => ({
            hull: hullBuilds[i],
            transform: placement(child.transform, sx, sy, sz),
            material: child.material ?? defaultSurfaceMaterial(),
        })),
        meshes: meshChildren.map((child, i) => {
            const materials = child.materials ?? [defaultSurfaceMaterial()];
            return {
                meshData: meshBuilds[i],
                transform: placement(child.transform, sx, sy, sz),
                scale: vec3(child.scale, sx, sy, sz),
                materials,
                materialCount: materials.length,
            };
        }),
        spheres: def.spheres?.map((child) => ({
            sphere: { center: vec3(child.center, sx, sy, sz), radius: child.radius * Math.abs(sx) },
            material: child.material ?? defaultSurfaceMaterial(),
        })),
    });
    if (!data) throw new Error(`${label} could not be built`);
    return { data, hulls: hullBuilds, meshes: meshBuilds };
}

const compounds: Geometry<Compound, CompoundBuild> = {
    registry: Compounds,
    builds: { create: () => new WeakMap() },
    scaled: { create: () => new WeakMap() },
    build: buildCompound,
    current: (world, def, built, sx = 1, sy = 1, sz = 1) =>
        (def.hulls ?? []).every(
            (child, i) => lookup(world, hulls, child.hull, sx, sy, sz) === built.hulls[i],
        ) &&
        (def.meshes ?? []).every(
            (child, i) => lookup(world, meshes, child.mesh) === built.meshes[i],
        ),
};

/** The built compound for a Compounds id, or undefined when none is registered; throws for a definition a solver refuses. */
export function compoundBuild(
    world: World,
    id: number,
    sx = 1,
    sy = 1,
    sz = 1,
): CompoundData | undefined {
    return lookup(world, compounds, id, sx, sy, sz)?.data;
}

/** Builds of the registered definitions by registry id, by reference; builds are immutable. */
export interface GeometryBuilds {
    hulls: [number, HullData, string?][];
    meshes: [number, MeshData, string?][];
    heightFields: [number, HeightFieldData, string?][];
    compounds: [number, CompoundBuild, string?][];
}

export function captureGeometryBuilds(world: World): GeometryBuilds {
    return {
        hulls: capture(world, hulls),
        meshes: capture(world, meshes),
        heightFields: capture(world, heightFields),
        compounds: capture(world, compounds),
    };
}

/** Pairs captured builds with the definitions core physics restored, so restored Shapes keep their geometry identity. */
export function restoreGeometryBuilds(world: World, captured: GeometryBuilds): void {
    restore(world, hulls, captured.hulls);
    restore(world, meshes, captured.meshes);
    restore(world, heightFields, captured.heightFields);
    restore(world, compounds, captured.compounds);
}
