import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { cube, Meshes, MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    PointLight,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation } from "../../engine";
import { StandardRenderer, StandardRenderingPlugin } from "./index";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";
import { PointShadows } from "./shadows";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin],
    },
]);

test("default diffuse rejects a shadowed point light behind a box side", async () => {
    const { world } = subjects()[0];
    world.resource(PointShadows).atlas = 1024;
    world.resource(PointShadows).casters = 1;
    const camera = world.create();
    world.add(camera, Transform, {
        translation: [-1, 0.55, 6],
        rotation: lookAtRotation(-1, 0.55, 6, -1, 0.55, 4),
    });
    world.add(camera, Camera, {
        mode: CameraMode.Perspective,
        fov: 45,
        far: 20,
        clearColor: 0,
        antialias: 0,
    });
    world.add(camera, AmbientLight, { brightness: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 256, height: 256 });
    const materials = world.resource(Materials);
    const material = materials.add(StandardMaterial());
    const box = world.create();
    world.add(box, Transform, { translation: [-1, 0.55, 4] });
    world.add(box, MeshInstance, {
        mesh: world.resource(Meshes).add({ name: "lambert-box", ...cube([0.8, 1.1, 0.8]) }),
    });
    world.add(box, MeshMaterial, material);
    const light = world.create();
    world.add(light, Transform, { translation: [-1, 3.85, 4] });
    world.add(light, PointLight, {
        intensity: 400000,
        range: 10,
        radius: 0.1,
        shadowMapsEnabled: 1,
        depthBias: 0.08,
        shadowNormalBias: 0.6,
    });
    async function edgeLight(): Promise<number> {
        world.step(0);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        let maximum = 0;
        // The front normal points away from the pendant.
        // Sample below the top edge at y=1.1, excluding the face's left and right edges.
        for (let y = 20; y < 45; y++) {
            for (let x = 100; x < 156; x++) maximum = Math.max(maximum, rgba[(y * 256 + x) * 4]);
        }
        return maximum;
    }
    expect(await edgeLight()).toBe(0);
});
