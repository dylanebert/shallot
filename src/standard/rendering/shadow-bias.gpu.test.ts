import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { cube, MeshInstance, registerMesh } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    DirectionalLight,
    PointLight,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation, type World } from "../../engine";
import { StandardRenderer, StandardRenderingPlugin } from "./index";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";
import { DirectionalLightShadowMap, PointShadows } from "./shadows";

setDefaultTimeout(CEILING.gpu);
const config = { defaults: false, plugins: [StandardRenderingPlugin, MeshRenderPlugin] };
const subjects = gpuApps(import.meta.path, [config, config]);
const WIDTH = 256;
const EDGE_ROW = 160;
const WORLD_WIDTH = 8;
const DEPTH_BIAS = 0.5;

type Kind = "directional" | "point";

function makeScene(world: World, kind: Kind): { camera: number; light: number } {
    world.resource(DirectionalLightShadowMap).size = 512;
    world.resource(PointShadows).atlas = 1024;
    world.resource(PointShadows).casters = 1;

    const camera = world.create();
    world.add(camera, Transform, {
        translation: [0, 30, 0, 0],
        rotation: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2],
    });
    world.add(camera, Camera, {
        mode: CameraMode.Orthographic,
        size: 4,
        far: 60,
        clearColor: 0,
        antialias: 0,
    });
    world.add(camera, AmbientLight, { brightness: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: WIDTH, height: WIDTH });

    const material = world.resource(Materials).add(
        StandardMaterial({
            baseColor: [1, 1, 1, 1],
            metallic: 0,
            perceptualRoughness: 1,
            diffuseWrap: 1,
        }),
    );
    const floor = world.create();
    world.add(floor, Transform, { translation: [0, -0.1, 0, 0] });
    world.add(floor, MeshInstance, {
        mesh: registerMesh(world, { name: `${kind}-bias-floor`, ...cube([10, 0.2, 10]) }),
    });
    world.add(floor, MeshMaterial, material);

    const blocker = world.create();
    world.add(blocker, Transform, { translation: [0, 0.75, 0, 0] });
    world.add(blocker, MeshInstance, {
        mesh: registerMesh(world, { name: `${kind}-bias-blocker`, ...cube([1, 1.5, 1]) }),
    });
    world.add(blocker, MeshMaterial, material);

    const light = world.create();
    if (kind === "directional") {
        const q = lookAtRotation(0, 0, 0, 0.5, -1, 0.5);
        world.add(light, Transform, { rotation: [q.x, q.y, q.z, q.w] });
        world.add(light, DirectionalLight, {
            illuminance: 10_000,
            shadowMapsEnabled: 1,
            maximumDistance: 50,
            numCascades: 1,
            firstCascadeFarBound: 8,
            overlapProportion: 0,
            depthBias: 0,
            shadowNormalBias: 0,
        });
    } else {
        world.add(light, Transform, { translation: [-10, 20, -10, 0] });
        world.add(light, PointLight, {
            intensity: 1_000_000,
            range: 40,
            radius: 0.1,
            shadowMapsEnabled: 1,
            depthBias: 0,
            shadowNormalBias: 0,
        });
    }
    return { camera, light };
}

async function edgeX(world: World, camera: number): Promise<number> {
    world.step(0);
    world.step(0);
    const { rgba } = await captureTexture(world, camera);
    let litReference = 0;
    for (let x = 40; x < 90; x++)
        litReference = Math.max(litReference, rgba[(EDGE_ROW * WIDTH + x) * 4]);
    const threshold = litReference * 0.1;
    for (let x = 80; x < 180; x++) {
        const left = rgba[(EDGE_ROW * WIDTH + x) * 4];
        const right = rgba[(EDGE_ROW * WIDTH + x + 1) * 4];
        if (left >= threshold && right < threshold) {
            return x + (left - threshold) / (left - right);
        }
    }
    throw new Error(`no shadow edge in row ${EDGE_ROW}; lit reference ${litReference}`);
}

test("equal directional and point depthBias values move production shadow edges by the same world distance", async () => {
    const [directionalSubject, pointSubject] = subjects();
    const directionalWorld = directionalSubject.world;
    const pointWorld = pointSubject.world;
    const directional = makeScene(directionalWorld, "directional");
    const point = makeScene(pointWorld, "point");

    const directionalStorage = directionalWorld.storage(DirectionalLight);
    const pointStorage = pointWorld.storage(PointLight);
    const directionalAtZero = await edgeX(directionalWorld, directional.camera);
    const pointAtZero = await edgeX(pointWorld, point.camera);
    directionalStorage.depthBias.set(directional.light, DEPTH_BIAS);
    pointStorage.depthBias.set(point.light, DEPTH_BIAS);
    const directionalBiased = await edgeX(directionalWorld, directional.camera);
    const pointBiased = await edgeX(pointWorld, point.camera);

    // The orthographic view spans eight world units across 256 pixels. The finite point source bends
    // the shadow boundary slightly relative to the parallel directional rays, so allow four pixels.
    const worldPerPixel = WORLD_WIDTH / WIDTH;
    const directionalShift = (directionalBiased - directionalAtZero) * worldPerPixel;
    const pointShift = (pointBiased - pointAtZero) * worldPerPixel;
    expect(directionalShift).toBeGreaterThan(0.1);
    expect(pointShift).toBeGreaterThan(0.03);
    expect(Math.abs(directionalShift - pointShift)).toBeLessThanOrEqual(4 * worldPerPixel);
});
