import { type Transform, type Vec3, xf } from "../common/math";
import { defaultBodyDef, defaultShapeDef, defaultWorldDef, ShapeType } from "../common/types";
import { rebuildGeometry } from "../kernel/geocolumns";
import { init, kernel } from "../kernel/kernel";
import type { CompoundData } from "../shapes/compound";
import type { Capsule, Sphere } from "../shapes/geometry";
import type { HeightFieldData } from "../shapes/heightfield";
import type { HullData } from "../shapes/hull";
import type { Mesh } from "../shapes/mesh";
import {
    createCapsuleShape,
    createCompoundShape,
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
} from "../shapes/shape";
import { createBody } from "../world/body";
import { createWorld, destroyWorld, getWorld, type WorldState } from "../world/world";
import type { CastOutput, RayCastInput, ShapeCastInput, ShapeProxy } from "./distance";
import type { PlaneResult } from "./mover";

await init(undefined, { threads: 0 });

type Geometry = Sphere | Capsule | HullData | Mesh | HeightFieldData | CompoundData;

function subject(
    kind: ShapeType,
    geometry: Geometry,
    run: (world: WorldState, id: number) => void,
): void {
    const world = getWorld(createWorld(undefined, defaultWorldDef())) as WorldState;
    try {
        const body = createBody(world, defaultBodyDef());
        const def = defaultShapeDef();
        const shape =
            kind === ShapeType.Sphere
                ? createSphereShape(world, body, def, geometry as Sphere)
                : kind === ShapeType.Capsule
                  ? createCapsuleShape(world, body, def, geometry as Capsule)
                  : kind === ShapeType.Hull
                    ? createHullShape(world, body, def, geometry as HullData)
                    : kind === ShapeType.Mesh
                      ? createMeshShape(
                            world,
                            body,
                            def,
                            (geometry as Mesh).data,
                            (geometry as Mesh).scale,
                        )
                      : kind === ShapeType.HeightField
                        ? createHeightFieldShape(world, body, def, geometry as HeightFieldData)
                        : createCompoundShape(world, body, def, geometry as CompoundData);
        if (!shape) throw new Error("gold shape creation failed");
        rebuildGeometry(world);
        run(world, shape.id);
    } finally {
        destroyWorld(world);
    }
}
function input(
    transform: Transform,
    proxy: ShapeProxy,
    translation: Vec3,
    fraction: number,
    encroach: boolean,
): void {
    const k = kernel(undefined);
    const r = new Float32Array(k.memory.buffer, k.shapeQueryInputPtr(), 398);
    r.fill(0);
    r.set([
        transform.p.x,
        transform.p.y,
        transform.p.z,
        transform.q.v.x,
        transform.q.v.y,
        transform.q.v.z,
        transform.q.s,
        proxy.count,
        proxy.radius,
        translation.x,
        translation.y,
        translation.z,
        fraction,
        Number(encroach),
    ]);
    for (let i = 0; i < proxy.count; ++i) {
        const p = proxy.points[i];
        r.set([p.x, p.y, p.z], 14 + 3 * i);
    }
}
function output(): CastOutput {
    const k = kernel(undefined);
    const r = new Float32Array(k.memory.buffer, k.shapeQueryOutputPtr(), 12);
    return {
        hit: r[0] !== 0,
        fraction: r[1],
        point: { x: r[2], y: r[3], z: r[4] },
        normal: { x: r[5], y: r[6], z: r[7] },
        iterations: r[8],
        triangleIndex: r[9],
        childIndex: r[10],
        materialIndex: r[11],
    };
}
export function kernelRay(kind: ShapeType, geometry: Geometry, ray: RayCastInput): CastOutput {
    let result: CastOutput | undefined;
    subject(kind, geometry, (world, id) => {
        input(
            xf.identity(),
            { points: [ray.origin], count: 1, radius: 0 },
            ray.translation,
            ray.maxFraction,
            false,
        );
        kernel(undefined).shapeQueryRay(world.worldId, id, 1);
        result = output();
    });
    return result as CastOutput;
}
export function kernelCast(kind: ShapeType, geometry: Geometry, cast: ShapeCastInput): CastOutput {
    let result: CastOutput | undefined;
    subject(kind, geometry, (world, id) => {
        result = kernelCastResident(world, id, cast);
    });
    return result as CastOutput;
}

/** The caller owns the shape and has made its world's columns and geometry resident. */
export function kernelCastResident(
    world: WorldState,
    id: number,
    cast: ShapeCastInput,
): CastOutput {
    input(xf.identity(), cast.proxy, cast.translation, cast.maxFraction, cast.canEncroach);
    kernel(undefined).shapeQueryCast(world.worldId, id, 1);
    return output();
}

export function kernelOverlap(
    kind: ShapeType,
    geometry: Geometry,
    transform: Transform,
    proxy: ShapeProxy,
): boolean {
    let result = false;
    subject(kind, geometry, (world, id) => {
        input(transform, proxy, { x: 0, y: 0, z: 0 }, 0, false);
        result = kernel(undefined).shapeQueryOverlap(world.worldId, id) !== 0;
    });
    return result;
}
export function kernelMover(
    kind: ShapeType,
    geometry: Geometry,
    mover: Capsule,
    capacity = 16,
): PlaneResult[] {
    const results: PlaneResult[] = [];
    subject(kind, geometry, (world, id) => {
        input(
            xf.identity(),
            { points: [mover.center1, mover.center2], count: 2, radius: mover.radius },
            { x: 0, y: 0, z: 0 },
            0,
            false,
        );
        const k = kernel(undefined);
        const ptr = k.scratchPtr();
        const count = k.shapeQueryMover(world.worldId, id, ptr, capacity, 1);
        const f = new Float32Array(k.memory.buffer, ptr, count * 10);
        for (let i = 0; i < count; ++i) {
            const o = i * 10;
            results.push({
                plane: { normal: { x: f[o], y: f[o + 1], z: f[o + 2] }, offset: f[o + 3] },
                point: { x: f[o + 4], y: f[o + 5], z: f[o + 6] },
            });
        }
    });
    return results;
}
