import { Body, PhysicsMeshes, Shape, ShapeKind, Transform, type World } from "@dylanebert/shallot";
import { kernel } from "../../src/standard/physics/kernel/kernel";
import { human } from "./creation-order-human";

const f = Math.fround;
// mesh.c's input vertices and indices, before either path builds the mesh tree.
export function rainGround(world: World): void {
    const vertices: number[] = [], indices: number[] = [];
    for (let ix = 0; ix <= 8; ix++)
        for (let iz = 0; iz <= 8; iz++) vertices.push(-7.5 + ix * 1.875, 0, -7.5 + iz * 1.875);
    for (let ix = 0; ix < 8; ix++) for (let iz = 0; iz < 8; iz++) {
        const a = iz + 9 * ix, b = a + 1, c = b + 9, d = c - 1;
        indices.push(a, b, c, c, d, a);
    }
    const meshes = world.resource(PhysicsMeshes);
    const grid = meshes.add({ name: "rain-grid", vertices: new Float32Array(vertices), indices: new Uint32Array(indices), materialIndices: new Uint8Array(128), useMedianSplit: true, identifyEdges: true });
    vertices.length = indices.length = 0;
    const k = kernel(world);
    const cs = (angle: number): [number, number] => {
        const p = k.mathCosSin(angle);
        const v = new Float32Array(k.memory.buffer, p, 2);
        return [v[0], v[1]];
    };
    for (let r = 0; r < 16; r++) for (let t = 0; t < 16; t++) {
        const [cu, su] = cs(f(f(t / 16) * f(2 * Math.PI)));
        const [cv, sv] = cs(f(f(r / 16) * f(2 * Math.PI)));
        vertices.push(f(f(3.75 + cv) * cu), f(f(3.75 + cv) * su), sv);
    }
    for (let r = 0; r < 16; r++) for (let t = 0; t < 16; t++) {
        const a = r * 16 + t, b = r * 16 + (t + 1) % 16;
        const c = ((r + 1) % 16) * 16 + (t + 1) % 16, d = ((r + 1) % 16) * 16 + t;
        indices.push(a, b, c, c, d, a);
    }
    const torus = meshes.add({ name: "rain-torus", vertices: new Float32Array(vertices), indices: new Uint32Array(indices), identifyEdges: true });
    for (let i = 0; i < 10; i++) for (let j = 0; j < 10; j++) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [-67.5 + 15 * i, 0, -67.5 + 15 * j] });
        world.add(eid, Body);
        world.add(eid, Shape, { kind: ShapeKind.Mesh, geometry: grid });
        const other = world.create();
        world.add(other, Shape, { body: eid, kind: ShapeKind.Mesh, geometry: torus });
    }
}

export function rainSpawner(): (world: World, step: number) => void {
    const groups = Array.from({ length: 100 }, () => [] as ReturnType<typeof human>[]);
    let columnCount = 0, columnIndex = 0;
    return (world, step) => {
        if ((step & 0x2f) !== 0) return;
        const filling = columnCount < 10;
        const column = filling ? columnCount++ : columnIndex;
        for (let row = 0; row < 10; row++) {
            const index = row * 10 + column;
            for (const group of groups[index]!) {
                world.destroy(group.joints.at(-1)!);
                for (const joint of group.joints.slice(0, -1)) world.destroy(joint);
                for (const body of group.bodies) world.destroy(body);
            }
            const next: ReturnType<typeof human>[] = [];
            let x = -75 + 15 * (column + 0.5);
            for (let i = 0; i < 3; i++, x = f(x + 0.75)) next.push(human(world, [x, 20, -75 + 15 * (row + 0.5)], index));
            groups[index] = next;
        }
        if (!filling) columnIndex = (columnIndex + 1) % 10;
    };
}
