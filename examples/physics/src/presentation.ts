import {
    cube,
    type Hull,
    Hulls,
    Shape,
    ShapeKind,
    shapeEdges,
    shapeMesh,
    type World,
} from "@dylanebert/shallot";
import { largeCompound } from "./compound-presentation";
import { controlValues, enabledControls, type Scene, scenes } from "./scenes";
import { drivenBodies, materials, shapeRole } from "./shape-color";
import { treeGroundHeight } from "./tree-presentation";
import { WorldSampleState } from "./world";

type Geometry = {
    mesh: ReturnType<typeof shapeMesh> & { name: string };
    edges: ReturnType<typeof shapeEdges>;
};
export type ShapeGeometryCache = Map<string, Geometry>;

// Hull definitions are immutable, so shapes naming equal hulls through distinct handles share one mesh.
const hullKeys = new WeakMap<Hull, string>();
function geometryKey(world: World, kind: number, geometry: number): unknown {
    const hull = kind === ShapeKind.Hull ? world.resource(Hulls).get(geometry) : undefined;
    if (!hull) return geometry;
    let key = hullKeys.get(hull);
    if (key === undefined) {
        const { name: _, points, ...rest } = hull;
        key = JSON.stringify([rest, points && Array.from(points)]);
        hullKeys.set(hull, key);
    }
    return key;
}

export function shapePresentation(
    world: World,
    eid: number,
    cache?: ShapeGeometryCache,
    driven: ReadonlySet<number> = drivenBodies(world),
) {
    const shape = world.get(eid, Shape)!;
    const body = shape.body || eid;
    const role = shapeRole(world, eid, driven);
    const key = JSON.stringify([
        shape.kind,
        geometryKey(world, shape.kind, shape.geometry),
        shape.scale,
        shape.radius,
        shape.sphereCenter,
        shape.capsuleCenter1,
        shape.capsuleCenter2,
        shape.hullPosition,
        shape.hullRotation,
        shape.enableHullTransform,
    ]);
    let geometry = cache?.get(key);
    if (!geometry) {
        geometry = {
            mesh: { name: `Shape ${eid}`, ...shapeMesh(world, shape) },
            edges: shapeEdges(world, shape),
        };
        cache?.set(key, geometry);
    }
    return { eid, body, role, ...geometry };
}

function explosionRings(
    center: [number, number, number],
    rings: readonly (readonly [number, number])[],
) {
    const lines: { a: [number, number, number]; b: [number, number, number]; color: number }[] = [];
    for (const [radius, color] of rings)
        for (let axis = 0; axis < 3; axis++)
            for (let i = 0; i < 24; i++) {
                const point = (index: number): [number, number, number] => {
                    const p: [number, number, number] = [...center];
                    const angle = (index * 2 * Math.PI) / 24;
                    p[(axis + 1) % 3]! += radius * Math.cos(angle);
                    p[(axis + 2) % 3]! += radius * Math.sin(angle);
                    return p;
                };
                lines.push({ a: point(i), b: point(i + 1), color });
            }
    return lines;
}

export function scenePresentation(
    world: World,
    scene: Scene,
    cache: ShapeGeometryCache = new Map(),
) {
    const driven = drivenBodies(world);
    const shapes = Array.from(scene === "largeWorld" ? [] : world.query([Shape]))
        .filter((eid) => !largeCompound(world, eid))
        .map((eid) => shapePresentation(world, eid, cache, driven));
    const definition = scenes[scene];
    const { gridExtent: extent, gridSpacing: spacing } = definition;
    const height =
        (scene === "treeBenchmark"
            ? treeGroundHeight(world)
            : "gridHeight" in definition
              ? definition.gridHeight
              : 0) + 0.005;
    const [cx, cz] =
        scene === "farStack"
            ? [world.resource(WorldSampleState).offset * 1000, 0]
            : "gridCenter" in definition
              ? definition.gridCenter
              : [0, 0];
    const lines: { a: [number, number, number]; b: [number, number, number]; color: number }[] = [];
    for (let i = 0; i <= Math.round((2 * extent) / spacing); i++) {
        const offset = -extent + i * spacing;
        const color = offset === 0 ? 0x797369 : 0x49453f;
        lines.push({
            a: [cx! - extent, height, cz! + offset],
            b: [cx! + extent, height, cz! + offset],
            color,
        });
    }
    for (let i = 0; i <= Math.round((2 * extent) / spacing); i++) {
        const offset = -extent + i * spacing;
        const color = offset === 0 ? 0x797369 : 0x49453f;
        lines.push({
            a: [cx! + offset, height, cz! - extent],
            b: [cx! + offset, height, cz! + extent],
            color,
        });
    }
    return {
        shapes,
        grid: { lines, width: 1, opacity: 0.3 },
        explosion:
            scene === "destruction"
                ? explosionRings(
                      [0, 5, 0],
                      [
                          [2.5, 0x00ffff],
                          [3.75, 0xfff8dc],
                      ],
                  )
                : scene === "weeble"
                  ? explosionRings([0, -0.1, 0], [[8, 0x00ffff]])
                  : [],
        ground:
            scene === "sensor"
                ? {
                      mesh: { name: "Ground", ...cube([20, 0.2, 20]) },
                      translation: [0, -0.1, 0] as [number, number, number],
                  }
                : undefined,
        materials,
        controls: enabledControls(scene, controlValues(scene)),
    };
}
