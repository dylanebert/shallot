import {
    Body,
    PhysicsMeshes,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    type World,
} from "@dylanebert/shallot";
import {
    createPhysicsMesh,
    destroyPhysicsMesh,
    type PhysicsMeshData,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";
import { decodeMesh } from "./mesh-data";
import type { Controls } from "./scenes";
import voxel01 from "./voxel_mesh_01.json" with { type: "json" };
import voxel02 from "./voxel_mesh_02.json" with { type: "json" };
import voxel03 from "./voxel_mesh_03.json" with { type: "json" };
import voxel04 from "./voxel_mesh_04.json" with { type: "json" };

const f = Math.fround;
// Box3D's LoadTempMesh at scale 0.01 with Z up.
const meshes = [voxel01, voxel02, voxel03, voxel04].map((encoded) => {
    const mesh = decodeMesh(encoded);
    const vertices = new Float32Array(mesh.vertices.length);
    for (let i = 0; i < vertices.length; i += 3) {
        vertices[i] = f(f(0.01) * mesh.vertices[i + 1]!);
        vertices[i + 1] = f(f(0.01) * mesh.vertices[i + 2]!);
        vertices[i + 2] = f(f(0.01) * mesh.vertices[i]!);
    }
    return { ...mesh, vertices };
});
const initial = {
    index: 0,
    Split: 1,
    "concave edges": 1,
    "weld vertices": 1,
    tolerance: 1.5,
    "draw level": -1,
};
export const MeshInspectorState = {
    create: () => ({
        active: false,
        body: 0,
        geometry: -1,
        values: { ...initial },
        report: null as PhysicsMeshData | null,
        area: 0,
        height: 0,
    }),
};
export const MeshCreationState = {
    create: () => ({ active: false, triangleCount: 0, time: Number.MAX_VALUE }),
};
const MeshCreationPlugin: Plugin = {
    name: "MeshCreation",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(MeshCreationState),
    systems: [
        {
            name: "mesh-creation-benchmark",
            group: "fixed",
            update(world) {
                const s = world.resource(MeshCreationState);
                if (!s.active) return;
                let triangleCount = 0;
                for (let i = 0; i < 10; i++) {
                    const start = performance.now();
                    for (const mesh of meshes) {
                        const report = createPhysicsMesh(world, {
                            name: "benchmark",
                            ...mesh,
                            useMedianSplit: true,
                            identifyEdges: false,
                            weldVertices: true,
                            weldTolerance: 0.0015,
                        });
                        if (i === 0) triangleCount += report.triangleCount;
                        destroyPhysicsMesh(world, report);
                    }
                    s.time = Math.min(s.time, performance.now() - start);
                }
                s.triangleCount = triangleCount;
            },
        },
    ],
};

export const MeshInspectorPlugin: Plugin = {
    name: "MeshInspector",
    dependencies: [StandardPhysicsPlugin, MeshCreationPlugin],
    recovery: recoverResource(MeshInspectorState),
};

// Box3D's sample_mesh.cpp, ComputeInternalSurfaceArea.
function internalSurfaceArea(mesh: PhysicsMeshData): number {
    let area = 0;
    for (let i = 1; i < mesh.nodes.length; i++) {
        const node = mesh.nodes[i]!;
        if (node.leaf) continue;
        const { lowerBound: a, upperBound: b } = node.bounds;
        const x = f(b.x - a.x),
            y = f(b.y - a.y),
            z = f(b.z - a.z);
        area = f(area + f(2 * f(f(f(x * y) + f(y * z)) + f(z * x))));
    }
    return area;
}
function loadMesh(world: World): void {
    const s = world.resource(MeshInspectorState);
    if (s.body) world.destroy(s.body);
    if (s.geometry >= 0) world.resource(PhysicsMeshes).remove(s.geometry);
    const definition = {
        name: "viewer",
        ...meshes[s.values.index]!,
        useMedianSplit: !!s.values.Split,
        identifyEdges: !!s.values["concave edges"],
        weldVertices: !!s.values["weld vertices"],
        weldTolerance: f(f(0.001) * f(s.values.tolerance)),
    };
    s.body = world.create();
    world.add(s.body, Body);
    s.report = createPhysicsMesh(world, definition);
    destroyPhysicsMesh(world, s.report);
    s.geometry = world.resource(PhysicsMeshes).add(definition);
    world.add(s.body, Shape, { kind: ShapeKind.Mesh, geometry: s.geometry });
    s.area = internalSurfaceArea(s.report);
    // b3GetHeight measures root-to-leaf edges, independently of b3MeshData's stored treeHeight.
    s.height = Math.max(0, s.report.nodeBounds.length - 1);
    s.values["draw level"] = Math.max(-1, Math.min(s.values["draw level"], s.height));
}
const controls: Controls = {
    index: { label: "index", value: 0, min: 0, max: 3, step: 1, decimals: 0, set: reload("index") },
    Split: {
        label: "Split",
        value: 1,
        select: true,
        options: [
            { label: "median split", value: 1 },
            { label: "sah binning", value: 0 },
        ],
        set: reload("Split"),
    },
    "concave edges": { label: "concave edges", value: 1, set: reload("concave edges") },
    "weld vertices": { label: "weld vertices", value: 1, set: reload("weld vertices") },
    tolerance: {
        label: "tolerance",
        value: 1.5,
        min: 0,
        max: 10,
        step: 0.1,
        decimals: 1,
        set: reload("tolerance"),
    },
    "draw level": {
        label: "draw level",
        value: -1,
        min: -1,
        max: 0,
        step: 1,
        decimals: 0,
        bounds: (_values, world) => [-1, world?.resource(MeshInspectorState).height ?? 0],
        set(world, _bodies, value) {
            world.resource(MeshInspectorState).values["draw level"] = value;
        },
    },
    ...Object.fromEntries(
        ["triangle count", "vertex count", "degenerate count", "node area", "height"].map(
            (label, index) => [
                label,
                {
                    label,
                    value: 0,
                    ...(index === 3 ? { precision: 6 } : { decimals: 0 }),
                    read(world: World) {
                        const s = world.resource(MeshInspectorState);
                        return s.report
                            ? [
                                  s.report.triangleCount,
                                  s.report.vertexCount,
                                  s.report.degenerateCount,
                                  s.area,
                                  s.height,
                              ][index]!
                            : null;
                    },
                    set() {},
                },
            ],
        ),
    ),
};
function reload(name: keyof typeof initial): Controls[string]["set"] {
    return (world, _bodies, value, values) => {
        const s = world.resource(MeshInspectorState);
        s.values[name] = value;
        loadMesh(world);
        values["draw level"] = s.values["draw level"];
    };
}
export const meshViewerScene = {
    construct(world: World): number[] {
        Object.assign(world.resource(MeshInspectorState), MeshInspectorState.create(), {
            active: true,
        });
        loadMesh(world);
        return [];
    },
    controls,
    gridExtent: 20,
    gridSpacing: 2,
    gridCenter: [16, 16],
    camera: { yaw: 35, pitch: 25, distance: 70, phoneDistance: 140, target: [16, 10, 16] },
};
export const meshCreationScene = {
    construct(world: World): number[] {
        Object.assign(world.resource(MeshCreationState), MeshCreationState.create(), {
            active: true,
        });
        world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
        return [];
    },
    controls: {
        "triangle count": {
            label: "triangle count",
            value: 0,
            decimals: 0,
            read: (world: World) => {
                const s = world.resource(MeshCreationState);
                return s.time === Number.MAX_VALUE ? null : s.triangleCount;
            },
            set() {},
        },
        "total time": {
            label: "total time, ms",
            value: 0,
            decimals: 4,
            read: (world: World) => {
                const time = world.resource(MeshCreationState).time;
                return time === Number.MAX_VALUE ? null : time;
            },
            set() {},
        },
        "time per mesh": {
            label: "time per mesh, ms",
            value: 0,
            decimals: 4,
            read: (world: World) => {
                const time = world.resource(MeshCreationState).time;
                return time === Number.MAX_VALUE ? null : time / 4;
            },
            set() {},
        },
    },
    gridExtent: 10,
    gridSpacing: 1,
    camera: { yaw: 35, pitch: 25, distance: 24, phoneDistance: 38, target: [0, 0, 0] },
};
