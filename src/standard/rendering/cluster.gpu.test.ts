import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import {
    attachCanvas,
    Camera,
    CameraMode,
    computeViewProj,
    PointLight,
    RenderContext,
    Views,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { invertMat4, type World } from "../../engine";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { probeBuffer } from "../../engine/runtime";
import {
    CLUSTER_COUNT,
    CLUSTER_X,
    CLUSTER_Y,
    CLUSTER_Z,
    Clusters,
    clusterCoord,
    clusterIndex,
    clusterView,
    LIGHT_GRID_OFFSET,
    LIGHT_INDICES_OFFSET,
    LIGHT_POOL,
    lightClusters,
    requestLightOverflow,
    sliceDepth,
} from "./cluster";
import { StandardRenderingPlugin } from "./index";
import { MAX_POINT_LIGHTS } from "./lighting";

setDefaultTimeout(CEILING.gpu);
if (typeof ResizeObserver === "undefined") {
    Object.assign(globalThis, {
        ResizeObserver: class {
            observe() {}
            unobserve() {}
            disconnect() {}
        },
    });
}

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [StandardRenderingPlugin] },
]);

interface LightInput {
    position: [number, number, number];
    range: number;
}
interface LightCase extends LightInput {
    eid: number;
}
interface CameraFrame {
    eid: number;
    slot: number;
    projection: ReturnType<typeof clusterView>;
    view: Float32Array;
}
interface SceneSnapshot {
    sourceCount: number;
    compactEids: Map<number, number>;
    totalExpected: number;
    allocated: number;
    dropped: number;
}

function cameraFrame(world: World, eid: number): CameraFrame {
    const viewSlot = world.resource(Views).get(eid)!;
    const aspect = viewSlot.width / viewSlot.height;
    const view = new Float32Array(16);
    computeViewProj(world, eid, aspect, new Float32Array(16), view);
    return {
        eid,
        slot: viewSlot.slot,
        projection: clusterView(world, eid, aspect),
        view,
    };
}

function transformPoint(
    matrix: Float32Array,
    point: [number, number, number],
): [number, number, number] {
    const [x, y, z] = point;
    const w = matrix[3]! * x + matrix[7]! * y + matrix[11]! * z + matrix[15]!;
    return [
        (matrix[0]! * x + matrix[4]! * y + matrix[8]! * z + matrix[12]!) / w,
        (matrix[1]! * x + matrix[5]! * y + matrix[9]! * z + matrix[13]!) / w,
        (matrix[2]! * x + matrix[6]! * y + matrix[10]! * z + matrix[14]!) / w,
    ];
}

function pointFromView(world: World, camera: number, point: [number, number, number]) {
    return transformPoint(invertMat4(cameraFrame(world, camera).view), point);
}

function positionFor(index: number): [number, number, number] {
    switch (index % 3) {
        case 0:
            return [(((index * 7) % 9) - 4) * 0.4, (((index * 5) % 7) - 3) * 0.3, 0];
        case 1:
            return [4.2, (((index * 3) % 5) - 2) * 0.25, 0];
        default:
            return [24 + (index % 7), 0, 0];
    }
}

function makeLights(count: number): LightInput[] {
    return Array.from({ length: count }, (_, i) => ({
        position: positionFor(i),
        range: 2 ** ((i % 3) - 1),
    }));
}

// lightClusters applies the fine sphere/AABB predicate to every froxel. The raster path first visits
// only the projected sphere rectangle, so clip those CPU hits to the same conservative screen bounds.
function rasterClusters(
    view: ReturnType<typeof clusterView>,
    center: [number, number, number],
    radius: number,
): number[] {
    const viewMin = [center[0] - radius, center[1] - radius, Math.min(center[2] - radius, -1e-5)];
    const viewMax = [center[0] + radius, center[1] + radius, Math.min(center[2] + radius, -1e-5)];
    const projected: [number, number][] = [
        [viewMin[0]!, viewMin[1]!, viewMin[2]!],
        [viewMin[0]!, viewMin[1]!, viewMax[2]!],
        [viewMax[0]!, viewMax[1]!, viewMin[2]!],
        [viewMax[0]!, viewMax[1]!, viewMax[2]!],
    ].map(([x, y, z]) =>
        view.perspective
            ? [x! / (-z! * view.halfW), y! / (-z! * view.halfH)]
            : [x! / view.halfW, y! / view.halfH],
    );
    const ndcMin = [
        Math.max(-1, Math.min(...projected.map((point) => point[0]))),
        Math.max(-1, Math.min(...projected.map((point) => point[1]))),
    ];
    const ndcMax = [
        Math.min(1, Math.max(...projected.map((point) => point[0]))),
        Math.min(1, Math.max(...projected.map((point) => point[1]))),
    ];
    const minX = Math.max(
        0,
        Math.min(CLUSTER_X - 1, Math.floor((ndcMin[0]! + 1) * 0.5 * CLUSTER_X)) - 1,
    );
    const maxX = Math.min(
        CLUSTER_X - 1,
        Math.min(CLUSTER_X - 1, Math.floor((ndcMax[0]! + 1) * 0.5 * CLUSTER_X)) + 1,
    );
    const minY = Math.max(
        0,
        Math.min(CLUSTER_Y - 1, Math.floor((ndcMin[1]! + 1) * 0.5 * CLUSTER_Y)) - 1,
    );
    const maxY = Math.min(
        CLUSTER_Y - 1,
        Math.min(CLUSTER_Y - 1, Math.floor((ndcMax[1]! + 1) * 0.5 * CLUSTER_Y)) + 1,
    );
    return lightClusters(view, center, radius).filter((cluster) => {
        const { x, y } = clusterCoord(cluster);
        return x >= minX && x <= maxX && y >= minY && y <= maxY;
    });
}

function edgeLights(world: World, perspectiveCamera: number, secondCamera: number): LightInput[] {
    const frame = cameraFrame(world, perspectiveCamera);
    const view = frame.projection;
    const tangentBoundary = sliceDepth(view, 12);
    const tangent = pointFromView(world, perspectiveCamera, [0, 0, -(tangentBoundary + 1)]);
    const depth = 10;
    const cornerX = view.halfW * depth * 0.94;
    const cornerY = view.halfH * depth * 0.94;
    return [
        { position: pointFromView(world, perspectiveCamera, [0, 0, -0.2]), range: 0.15 },
        { position: pointFromView(world, perspectiveCamera, [0, 0, -999]), range: 2 },
        { position: pointFromView(world, perspectiveCamera, [0, 0, 2]), range: 0.5 },
        {
            position: pointFromView(world, perspectiveCamera, [cornerX, cornerY, -depth]),
            range: 0.7,
        },
        {
            position: pointFromView(world, perspectiveCamera, [-cornerX, -cornerY, -depth]),
            range: 0.7,
        },
        { position: pointFromView(world, secondCamera, [0, 0, -5]), range: 2 },
        { position: tangent, range: 1 },
    ];
}

async function checkScene(
    world: World,
    cameras: number[],
    inputs: LightInput[],
    verifyMembership = true,
): Promise<SceneSnapshot> {
    const cases: LightCase[] = [];
    for (const input of inputs) {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, PointLight);
        world.storage(Transform).translation.set(eid, ...input.position, 0);
        world.storage(PointLight).range.set(eid, input.range);
        cases.push({ eid, ...input });
    }

    try {
        world.step(1 / 60);
        expect(world.resource(RenderContext).shadeCount).toBe(2);
        await world.gpu.device.queue.onSubmittedWorkDone();
        const output = await probeBuffer(world, world.gpu.buffers.get("lightClusters")!, {
            size: LIGHT_INDICES_OFFSET + (LIGHT_POOL + 2) * 4,
            label: `cluster oracle ${inputs.length}`,
        });
        const bytes = output.bytes;
        const words = new Uint32Array(bytes);
        const sourceCount = words[0]!;
        const compactCount = Math.min(sourceCount, MAX_POINT_LIGHTS);
        const compactEids = new Map<number, number>();
        const data = new DataView(bytes);
        const compactOffset = 16;
        for (let light = 0; light < compactCount; light++) {
            compactEids.set(light, data.getFloat32(compactOffset + light * 48 + 28, true));
        }

        const compactedEids = new Set(compactEids.values());
        let totalExpected = 0;
        for (const camera of cameras) {
            const frame = cameraFrame(world, camera);
            const expected = verifyMembership
                ? Array.from({ length: CLUSTER_COUNT }, () => new Set<number>())
                : undefined;
            for (const light of cases) {
                if (!compactedEids.has(light.eid)) continue;
                const center = transformPoint(frame.view, light.position);
                const clusters = rasterClusters(frame.projection, center, light.range);
                totalExpected += clusters.length;
                if (expected) for (const cluster of clusters) expected[cluster]!.add(light.eid);
            }
            if (!expected) continue;
            const actual = Array.from({ length: CLUSTER_COUNT }, (_, cluster) => {
                const gridAt = LIGHT_GRID_OFFSET / 4 + (frame.slot * CLUSTER_COUNT + cluster) * 2;
                const start = words[gridAt]!;
                const length = words[gridAt + 1]!;
                const list: number[] = [];
                for (let i = 0; i < length; i++) {
                    const compact = words[LIGHT_INDICES_OFFSET / 4 + start + i]!;
                    const eid = compactEids.get(compact);
                    if (eid === undefined)
                        throw new Error(
                            `cluster ${cluster} references missing compact light ${compact}`,
                        );
                    list.push(eid);
                }
                return [...new Set(list)].sort((a, b) => a - b);
            });
            const sortedExpected = expected.map((list) => [...list].sort((a, b) => a - b));
            expect(actual).toEqual(sortedExpected);
        }

        const allocated = words[LIGHT_INDICES_OFFSET / 4]!;
        const dropped = words[LIGHT_INDICES_OFFSET / 4 + 1]!;
        if (verifyMembership) {
            expect(allocated).toBe(totalExpected);
            expect(dropped).toBe(0);
        }
        return {
            sourceCount,
            compactEids,
            totalExpected,
            allocated,
            dropped,
        };
    } finally {
        for (const light of cases) world.destroy(light.eid);
    }
}

test("GPU light clusters match the CPU oracle across projection, overflow, reset, and aborted draws", async () => {
    const app = subjects()[0]!;
    try {
        const world = app.world;
        const cameras: number[] = [];
        for (const [index, mode] of [CameraMode.Perspective, CameraMode.Orthographic].entries()) {
            const canvas = {
                width: 32,
                height: 24,
                style: { imageRendering: "auto" },
                getContext: () => context,
                getBoundingClientRect: () => ({ width: 32, height: 24 }),
            } as unknown as HTMLCanvasElement;
            const context = new CanvasContext(canvas, 32, 24);
            const camera = world.create();
            cameras.push(camera);
            world.add(camera, Transform);
            world.add(camera, Camera, { mode });
            world
                .storage(Transform)
                .translation.set(
                    camera,
                    index === 0 ? 0 : 8,
                    index === 0 ? 0 : -3,
                    index === 0 ? 5 : 12,
                    0,
                );
            if (index === 1) {
                const halfYaw = Math.sin(Math.PI / 8);
                world.storage(Transform).rotation.set(camera, 0, halfYaw, 0, Math.cos(Math.PI / 8));
            }
            attachCanvas(camera, canvas, world);
        }
        const [perspectiveCamera, secondCamera] = cameras as [number, number];

        // Establish camera GlobalTransforms and the first cluster grid before constructing view-space cases.
        world.step(1 / 60);
        expect(world.resource(RenderContext).shadeCount).toBe(2);
        const tangentView = clusterView(world, perspectiveCamera, 32 / 24);
        const tangentCenter: [number, number, number] = [0, 0, -(sliceDepth(tangentView, 12) + 1)];
        expect(lightClusters(tangentView, tangentCenter, 1)).toContain(clusterIndex(7, 4, 11));
        expect(rasterClusters(tangentView, tangentCenter, 1)).toContain(1715);
        expect(clusterCoord(1715)).toEqual({ x: 7, y: 4, z: 11 });
        expect(cameraFrame(world, perspectiveCamera).view).not.toEqual(
            cameraFrame(world, secondCamera).view,
        );

        await checkScene(world, cameras, []);
        const edges = edgeLights(world, perspectiveCamera, secondCamera);
        const perspective = cameraFrame(world, perspectiveCamera);
        const edgeCenters = edges.map((light) => transformPoint(perspective.view, light.position));
        expect(
            rasterClusters(perspective.projection, edgeCenters[0]!, edges[0]!.range).some(
                (cluster) => clusterCoord(cluster).z === 0,
            ),
        ).toBe(true);
        expect(
            rasterClusters(perspective.projection, edgeCenters[1]!, edges[1]!.range).some(
                (cluster) => clusterCoord(cluster).z === CLUSTER_Z - 1,
            ),
        ).toBe(true);
        expect(rasterClusters(perspective.projection, edgeCenters[2]!, edges[2]!.range)).toEqual(
            [],
        );
        const topRight = edgeCenters[3]!;
        const bottomLeft = edgeCenters[4]!;
        expect(
            rasterClusters(perspective.projection, topRight, 0.7).some((cluster) => {
                const { x, y } = clusterCoord(cluster);
                return x === CLUSTER_X - 1 && y === CLUSTER_Y - 1;
            }),
        ).toBe(true);
        expect(
            rasterClusters(perspective.projection, bottomLeft, 0.7).some((cluster) => {
                const { x, y } = clusterCoord(cluster);
                return x === 0 && y === 0;
            }),
        ).toBe(true);
        await checkScene(world, cameras, edges);
        for (const count of [1, 64, 256]) await checkScene(world, cameras, makeLights(count));

        // A populated-to-zero transition must clear every grid entry and both pool-header words.
        const empty = await checkScene(world, cameras, []);
        expect(empty.sourceCount).toBe(0);
        expect(empty.allocated).toBe(0);
        expect(empty.dropped).toBe(0);

        const overflowWarn = spyOn(console, "warn").mockImplementation(() => {});
        let overflow!: SceneSnapshot;
        try {
            overflow = await checkScene(
                world,
                cameras,
                Array.from({ length: 300 }, () => ({
                    position: [0, 0, 0] as [number, number, number],
                    range: 100000,
                })),
                false,
            );
            expect(overflowWarn).toHaveBeenCalledWith(
                `shallot: 300 point lights exceed the ${MAX_POINT_LIGHTS} cap; ${300 - MAX_POINT_LIGHTS} ignored`,
            );
        } finally {
            overflowWarn.mockRestore();
        }
        expect(overflow.sourceCount).toBe(300);
        expect(overflow.compactEids.size).toBe(MAX_POINT_LIGHTS);
        expect(overflow.allocated).toBe(LIGHT_POOL);
        expect(overflow.dropped).toBeGreaterThan(0);
        expect(overflow.allocated + overflow.dropped).toBe(overflow.totalExpected);
        expect((await requestLightOverflow(world)).dropped).toBe(overflow.dropped);
        const cleared = await checkScene(world, cameras, []);
        expect(cleared.sourceCount).toBe(0);
        expect(cleared.allocated).toBe(0);
        expect(cleared.dropped).toBe(0);

        // A propagated draw failure discards the encoded AABB rebuild. Its dirty key must remain old
        // until the same projection is successfully submitted on the following frame.
        const aabbs = world.gpu.buffers.get("clusterAabbs")!;
        const before = await probeBuffer(world, aabbs, {
            size: CLUSTER_COUNT * 2 * 16,
            label: "cluster grid before aborted projection",
        });
        const clusters = world.resource(Clusters);
        const committed = clusters.last.slice(0, 8);
        world.storage(Camera).fov.set(perspectiveCamera, 90);
        let abort = true;
        world.addSystem({
            group: "draw",
            last: true,
            update() {
                if (abort) {
                    abort = false;
                    throw new Error("abort projection rebuild");
                }
            },
        });
        const frame = world.gpu.frame;
        expect(() => world.step(0)).toThrow("abort projection rebuild");
        expect(world.gpu.frame).toBe(frame);
        expect(clusters.last.slice(0, 8)).toEqual(committed);
        await world.gpu.device.queue.onSubmittedWorkDone();
        const aborted = await probeBuffer(world, aabbs, {
            size: CLUSTER_COUNT * 2 * 16,
            label: "cluster grid after aborted projection",
        });
        expect(new Uint8Array(aborted.bytes)).toEqual(new Uint8Array(before.bytes));

        world.step(0);
        await world.gpu.device.queue.onSubmittedWorkDone();
        const rebuilt = await probeBuffer(world, aabbs, {
            size: CLUSTER_COUNT * 2 * 16,
            label: "cluster grid after retried projection",
        });
        expect(new Uint8Array(rebuilt.bytes)).not.toEqual(new Uint8Array(before.bytes));
        world.step(0);
        // Drain the final retry before app disposal releases the shared device.
        await world.gpu.device.queue.onSubmittedWorkDone();
        expect(clusters.last.slice(0, 8)).not.toEqual(committed);
    } finally {
        app.dispose();
    }
});
