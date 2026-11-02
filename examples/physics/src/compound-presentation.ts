import { Compounds, qRotate, Shape, ShapeKind, Transform, type World } from "@dylanebert/shallot";
import { shapeEdges, shapeMesh } from "@dylanebert/shallot/standard/physics";

type Point = [number, number, number];
type Rotation = [number, number, number, number];
type ChildShape = Parameters<typeof shapeMesh>[1];
export type CompoundChild = {
    shape: ChildShape;
    key: string;
    translation: Point;
    rotation: Rotation;
    scale: Point;
};
export function largeCompound(world: World, eid: number): boolean {
    const shape = world.storage(Shape);
    if (shape.kind.get(eid) !== ShapeKind.Compound) return false;
    const c = world.resource(Compounds).get(shape.geometry.get(eid))!;
    return (
        (c.capsules?.length ?? 0) +
            (c.hulls?.length ?? 0) +
            (c.meshes?.length ?? 0) +
            (c.spheres?.length ?? 0) >
        2500
    );
}
const identity: Rotation = [0, 0, 0, 1];
function multiply(a: Rotation, b: Rotation): Rotation {
    return [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ];
}
export function compoundChildren(world: World, eid: number): CompoundChild[] {
    const parent = world.get(eid, Shape)!;
    const owner = world.get(parent.body || eid, Transform)!;
    const compound = world.resource(Compounds).get(parent.geometry)!;
    const result: CompoundChild[] = [];
    const base = { ...parent, scale: [1, 1, 1] as Point };
    const add = (
        shape: ChildShape,
        key: string,
        position: Point,
        rotation: Rotation = identity,
        scale: Point = [1, 1, 1],
    ) => {
        const offset = qRotate(
            ...owner.rotation,
            ...(position.map((p, i) => p * owner.scale[i]!) as Point),
        );
        const translation = offset.map((p, i) => p + owner.translation[i]!) as Point;
        const q = multiply(owner.rotation, rotation);
        const size = scale.map((v, i) => v * owner.scale[i]!) as Point;
        result.push({
            shape,
            key,
            translation,
            rotation: q,
            scale: size,
        });
    };
    for (const c of compound.capsules ?? []) {
        const r = c.radius;
        const center = c.center1.map((x, i) => (x + c.center2[i]!) / 2) as Point;
        const a = c.center1.map((x, i) => x - center[i]!) as Point;
        const b = c.center2.map((x, i) => x - center[i]!) as Point;
        add(
            { ...base, kind: ShapeKind.Capsule, capsuleCenter1: a, capsuleCenter2: b, radius: r },
            JSON.stringify(["capsule", a, b, r]),
            center,
            identity,
            [1, 1, 1],
        );
    }
    for (const c of compound.hulls ?? [])
        add(
            { ...base, kind: ShapeKind.Hull, geometry: c.hull, enableHullTransform: 0 },
            `hull:${c.hull}`,
            c.transform.position,
            c.transform.rotation,
        );
    for (const c of compound.meshes ?? [])
        add(
            { ...base, kind: ShapeKind.Mesh, geometry: c.mesh },
            `mesh:${c.mesh}`,
            c.transform.position,
            c.transform.rotation,
            c.scale,
        );
    for (const c of compound.spheres ?? [])
        add(
            { ...base, kind: ShapeKind.Sphere, sphereCenter: [0, 0, 0], radius: 1 },
            "sphere",
            c.center,
            identity,
            [c.radius, c.radius, c.radius],
        );
    return result;
}
export function childGeometry(world: World, child: CompoundChild) {
    return {
        mesh: { name: child.key, ...shapeMesh(world, child.shape) },
        edges: shapeEdges(world, child.shape),
    };
}
