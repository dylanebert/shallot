import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { readFileSync } from "node:fs";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import {
    AmbientLight,
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    DirectionalLight,
    Exposure,
    ImageBindings,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation } from "../../engine/utils";
import {
    MeshRenderPlugin,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { brdf, Pbr } from "../../standard/rendering/shade";
import { loadGltfScene, poseGltfScene, spawnGltfScene } from "./scene";
import { GltfPlugin } from "./skinning";

setDefaultTimeout(TIMEOUT.gpu);
const light = std.normalize(d.vec3f(0.7, 0.3, 1));
let camera = 0;
let instance: ReturnType<typeof spawnGltfScene>;
let animation: Parameters<typeof poseGltfScene>[2];
const apps = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            GltfPlugin,
            {
                name: "SkinnedNormalScene",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin, GltfPlugin],
                async initialize(world) {
                    const sun = world.create();
                    world.add(sun, Transform, {
                        rotation: lookAtRotation(0, 0, 0, -light.x, -light.y, -light.z),
                    });
                    world.add(sun, DirectionalLight, { illuminance: 1.2, shadowMapsEnabled: 0 });
                    camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 4] });
                    world.add(camera, Camera, {
                        mode: CameraMode.Orthographic,
                        size: 1,
                        antialias: 0,
                    });
                    world.add(camera, StandardRenderer);
                    world.add(camera, Exposure, { ev100: 0 });
                    world.add(camera, AmbientLight, { brightness: 0.24 });
                    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
                    attachTexture(world, camera, { width: 16, height: 16 });
                    const bytes = readFileSync(
                        new URL(
                            "../../../examples/gltf/public/models/SkinnedNormal.gltf",
                            import.meta.url,
                        ),
                    );
                    const scene = await loadGltfScene(
                        world,
                        `data:model/gltf+json;base64,${bytes.toString("base64")}`,
                    );
                    animation = scene.animations[0];
                    instance = spawnGltfScene(world, scene, { scale: [1.2, 0.8, 1] });
                },
            },
        ],
    },
]);

beforeAll(() => {
    const { world } = apps()[0];
    // Dawn has no bitmap decoder: preserve the importer's normal-texture registration and
    // replace only decoding with the fixture PNG's single RGBA texel.
    const texture = world.gpu.device.createTexture({
        size: [1, 1, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(texture);
    world.gpu.device.queue.writeTexture(
        { texture },
        new Uint8Array([204, 80, 224, 255]),
        { bytesPerRow: 4 },
        [1, 1, 1],
    );
    world.resource(ImageBindings).dataImages = texture;
});

const display = (v: number) =>
    Math.round(Math.min(1, v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055) * 255);

// The fixture rotates its only joint from 0 to 120 degrees over [0,1]. Its nonuniform
// joint scale precedes rotation, and nonuniform spawn scale follows it. TANGENT is
// (sqrt(1/2),sqrt(1/2),0,-1), so inverse-transpose tangent deformation is distinguishable.
for (const time of [0.5, 1])
    test(`normal-mapped skin at recorded time ${time}: shaded texel equals CPU shade.ts with the posed tangent`, async () => {
        const { world } = apps()[0];
        poseGltfScene(world, instance, animation, time);
        world.tick();
        world.step(0);
        world.step(0);
        const angle = (time * 2 * Math.PI) / 3;
        const tangent = std.normalize(
            d.vec3f(
                1.2 * (1.5 * Math.cos(angle) - 0.75 * Math.sin(angle)),
                0.8 * (1.5 * Math.sin(angle) + 0.75 * Math.cos(angle)),
                0,
            ),
        );
        const mx = ((2 * 204) / 255 - 1) * 0.5;
        const my = ((2 * 80) / 255 - 1) * 0.5;
        const mz = (2 * 224) / 255 - 1;
        // N=(0,0,1), B=cross(N,T)*-1; normalize only the complete mapped normal.
        const normal = std.normalize(
            d.vec3f(tangent.x * mx + tangent.y * my, tangent.y * mx - tangent.x * my, mz),
        );
        // Orthographic pixel (8,8) is world (1/16,-1/16,0), camera at (0,0,4).
        const view = std.normalize(d.vec3f(-1 / 16, 1 / 16, 4));
        const direct = brdf(
            Pbr({
                albedo: d.vec3f(0.3, 0.4, 0.2),
                metallic: 0.4,
                roughness: 0.9,
                occlusion: 1,
                dielectric: 0.04,
            }),
            normal,
            view,
            light,
        );
        const expected = [...direct].map((v, i) => display(v + [0.3, 0.4, 0.2][i] * 0.2));
        const image = await captureTexture(world, camera);
        const actual = [...image.rgba.subarray((8 * 16 + 8) * 4, (8 * 16 + 8) * 4 + 3)];
        // Two 8-bit display steps admit float32 shading and framebuffer conversion, not a basis change.
        for (let i = 0; i < 3; i++)
            expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(2);
    });
