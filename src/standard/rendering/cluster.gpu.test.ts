import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import {
    attachCanvas,
    Camera,
    CameraMode,
    PointLight,
    RenderContext,
    Views,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { probeBuffer } from "../../engine/runtime";
import {
    CLUSTER_COUNT,
    clusterView,
    LIGHT_GRID_OFFSET,
    LIGHT_INDICES_OFFSET,
    LIGHT_POOL,
    lightClusters,
} from "./cluster";
import { StandardRenderingPlugin } from "./index";

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

const cameraIds: number[] = [];
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [StandardRenderingPlugin] },
]);

interface LightCase {
    eid: number;
    position: [number, number, number];
    range: number;
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

async function checkScene(world: World, count: number): Promise<void> {
    const cases: LightCase[] = [];
    for (let i = 0; i < count; i++) {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, PointLight);
        const position = positionFor(i);
        const range = 2 ** ((i % 3) - 1);
        world.storage(Transform).translation.set(eid, ...position, 0);
        world.storage(PointLight).range.set(eid, range);
        cases.push({ eid, position, range });
    }

    world.step(1 / 60);
    expect(world.resource(RenderContext).shadeCount).toBe(2);
    await world.gpu.device.queue.onSubmittedWorkDone();
    const output = await probeBuffer(world, world.gpu.buffers.get("lightClusters")!, {
        size: LIGHT_INDICES_OFFSET + (LIGHT_POOL + 2) * 4,
        label: `cluster oracle ${count}`,
    });
    const bytes = output.bytes;
    const words = new Uint32Array(bytes);
    const compactCount = Math.min(words[0]!, 256);
    const compactEids = new Map<number, number>();
    const compactOffset = 16;
    for (let light = 0; light < compactCount; light++) {
        compactEids.set(
            light,
            new DataView(bytes).getFloat32(compactOffset + light * 48 + 28, true),
        );
    }

    const views = world.resource(Views);
    for (const camera of cameraIds) {
        const viewSlot = views.get(camera)!.slot;
        const projection = clusterView(world, camera, 32 / 24);
        const expected = Array.from({ length: CLUSTER_COUNT }, () => new Set<number>());
        for (const light of cases) {
            const center: [number, number, number] = [
                light.position[0],
                light.position[1],
                light.position[2] - 5,
            ];
            for (const cluster of lightClusters(projection, center, light.range))
                expected[cluster]!.add(light.eid);
        }
        const actual = Array.from({ length: CLUSTER_COUNT }, (_, cluster) => {
            const gridAt = LIGHT_GRID_OFFSET / 4 + (viewSlot * CLUSTER_COUNT + cluster) * 2;
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
        expect(actual).toEqual(expected.map((list) => [...list].sort((a, b) => a - b)));
    }
    for (const light of cases) world.destroy(light.eid);
}

test("GPU light lists equal lightClusters for empty, interior, straddling and outside scenes in two view slots", async () => {
    const app = subjects()[0]!;
    try {
        const world = app.world;
        for (const mode of [CameraMode.Perspective, CameraMode.Orthographic]) {
            const canvas = {
                width: 32,
                height: 24,
                style: { imageRendering: "auto" },
                getContext: () => context,
                getBoundingClientRect: () => ({ width: 32, height: 24 }),
            } as unknown as HTMLCanvasElement;
            const context = new CanvasContext(canvas, 32, 24);
            const camera = world.create();
            cameraIds.push(camera);
            world.add(camera, Transform);
            world.add(camera, Camera, { mode });
            world.storage(Transform).translation.set(camera, 0, 0, 5, 0);
            attachCanvas(camera, canvas, world);
        }
        for (const count of [0, 1, 64, 256]) await checkScene(world, count);
    } finally {
        app.dispose();
    }
});
