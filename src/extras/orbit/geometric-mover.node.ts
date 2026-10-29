import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    GlobalTransform,
    globalTransformTable,
    pressKey,
    probeBuffer,
    RenderingPlugin,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import { attachCanvas } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import {
    GeometricMoverPlugin,
    GeometricMoverState,
} from "../../../examples/physics/src/geometric-mover";
import { createScene } from "../../../examples/physics/src/scenes";
import { setSceneCamera } from "../../../examples/physics/src/view";
import { createTestGpuApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { CanvasContext } from "../../engine/app/canvas.fixture";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();

test("Geometric Mover's GPU-drawn capsule and Orbit follow camera advance together on every 120 Hz frame", async () => {
    let camera = 0;
    const app = await createTestGpuApp({
        defaults: false,
        plugins: [GeometricMoverPlugin, OrbitPlugin, RenderingPlugin],
        setup(world) {
            createScene(world, "geometricMover");
            camera = world.create();
            world.add(camera, Camera);
            world.add(camera, Transform);
            world.add(camera, Orbit);
            setSceneCamera(world, "geometricMover", camera);
        },
    });
    try {
        const world = app.world;
        let context: CanvasContext;
        const canvas = {
            width: 32,
            height: 24,
            style: { imageRendering: "auto" },
            getContext: () => context,
            getBoundingClientRect: () => ({ width: 32, height: 24 }),
        } as unknown as HTMLCanvasElement;
        context = new CanvasContext(canvas, 32, 24);
        attachCanvas(world, camera, canvas);
        const state = world.resource(GeometricMoverState);
        const table = globalTransformTable(world);
        pressKey(world, "KeyW");
        let previousDrawn = 0,
            previousCamera = 0;
        for (let i = 0; i < 14; i++) {
            world.step(Time.FIXED_DT / 2);
            const result = await probeBuffer(world, table.buffer, {
                size: table.buffer.size,
                label: "mover-120hz-placement",
            });
            const drawn = new Float32Array(result.bytes)[table.rowIndex(state.entity) * 12 + 2]!;
            const followed = world.get(camera, GlobalTransform)!.translation[2];
            if (i >= 6) {
                expect(drawn - previousDrawn).toBeCloseTo(-0.05, 5);
                expect(followed - previousCamera).toBeCloseTo(drawn - previousDrawn, 5);
            }
            previousDrawn = drawn;
            previousCamera = followed;
        }
    } finally {
        await disposeGpuApps([app]);
    }
});
