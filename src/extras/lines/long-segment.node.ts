import { expect, setDefaultTimeout, test } from "bun:test";
import {
    cube,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    StandardMaterial,
    StandardRenderingPlugin,
    Transform,
} from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { attachTexture, Camera, captureTexture, DebandDither } from "../../core/rendering";
import { lookAtRotation } from "../../engine/utils";
import { drawLine, Gizmo, GizmoAsset, LinesPlugin, registerGizmo } from "./index";

setDefaultTimeout(TIMEOUT.node);
const subject = gpuScene(
    import.meta.path,
    { defaults: false, plugins: [StandardRenderingPlugin, LinesPlugin] },
    (world) => {
        const camera = world.create();
        world.add(camera, Transform, {
            translation: [0, 3, 6],
            rotation: lookAtRotation(0, 3, 6, 0, 0, 0),
        });
        world.add(camera, Camera, { clearColor: 0, antialias: 0 });
        world.add(camera, DebandDither, { enabled: 0 });
        attachTexture(world, camera, { width: 128, height: 128 });
        const floor = world.create();
        world.add(floor, Transform, { translation: [0, -0.1, 0] });
        world.add(floor, MeshInstance, {
            mesh: world.resource(Meshes).add({ name: "floor", ...cube([40, 0.2, 40]) }),
        });
        world.add(
            floor,
            MeshMaterial,
            world
                .resource(Materials)
                .add(StandardMaterial({ baseColor: [0.02, 0.02, 0.02, 1], unlit: true })),
        );
        const line = world.create();
        world.add(line, Transform);
        return { camera, line };
    },
);
test("a long ground line crossing the eye plane remains above the floor", async () => {
    const {
        world,
        scene: { camera, line },
    } = subject();
    console.info("long-line adapter:", world.gpu.adapter);
    for (const extent of [40, -40, 3300, -3300]) {
        const asset = new GizmoAsset();
        drawLine(asset, [1, 0.005, -extent], [1, 0.005, extent], 0xffffff);
        world.add(line, Gizmo, { handle: registerGizmo(world, asset), width: 3 });
        world.step(1 / 60);
        world.step(1 / 60);
        const { rgba } = await captureTexture(world, camera);
        let white = 0;
        for (let y = 75; y < 120; y++)
            for (let x = 0; x < 128; x++) {
                const i = (y * 128 + x) * 4;
                if (rgba[i]! > 180 && rgba[i + 1]! > 180 && rgba[i + 2]! > 180) white++;
            }
        console.log(`extent ${extent}: ${white} visible line pixels`);
        expect(white).toBeGreaterThan(20);
    }
});
