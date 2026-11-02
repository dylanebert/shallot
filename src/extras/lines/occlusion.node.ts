import { expect, setDefaultTimeout, test } from "bun:test";
import {
    cube,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
    Transform,
} from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { attachTexture, Camera, captureTexture, DebandDither } from "../../core/rendering";
import { drawLine, Gizmo, GizmoAsset, LinesPlugin, registerGizmo } from "./index";

setDefaultTimeout(TIMEOUT.node);
const subject = gpuScene(
    import.meta.path,
    { defaults: false, plugins: [StandardRenderingPlugin, MeshRenderPlugin, LinesPlugin] },
    (world) => {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5] });
        world.add(camera, Camera, { clearColor: 0, antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, DebandDither, { enabled: 0 });
        attachTexture(world, camera, { width: 128, height: 128 });
        const wall = world.create();
        world.add(wall, Transform);
        world.add(wall, MeshInstance, {
            mesh: world.resource(Meshes).add({ name: "occluder", ...cube([2, 2, 1]) }),
        });
        world.add(
            wall,
            MeshMaterial,
            world
                .resource(Materials)
                .add(StandardMaterial({ baseColor: [0.1, 0.1, 0.1, 1], unlit: true })),
        );
        const asset = new GizmoAsset();
        drawLine(asset, [-4, 0, -1], [4, 0, -1], 0xff0000);
        const line = world.create();
        world.add(line, Transform);
        world.add(line, Gizmo, {
            handle: registerGizmo(world, asset),
            width: 3,
            opacity: 0.45,
            depthBias: -0.0001,
        });
        return { camera };
    },
);
test("opaque geometry hides retained edges with the physics example's bias and opacity", async () => {
    const {
        world,
        scene: { camera },
    } = subject();
    world.step(1 / 60);
    world.step(1 / 60);
    const { rgba } = await captureTexture(world, camera);
    function redPixels(left: number, right: number) {
        let count = 0;
        for (let y = 60; y < 68; y++)
            for (let x = left; x < right; x++) {
                const i = (y * 128 + x) * 4;
                if (rgba[i]! > rgba[i + 1]! + 30) count++;
            }
        return count;
    }
    expect(redPixels(55, 73)).toBe(0);
    expect(redPixels(5, 30)).toBeGreaterThan(20);
});
