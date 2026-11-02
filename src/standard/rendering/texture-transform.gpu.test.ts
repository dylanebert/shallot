import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    Exposure,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { imageTextures } from "../../core/rendering/image";
import { Transform } from "../../core/transform";
import {
    extendedMaterial,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    pbrInputFromStandardMaterial,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const slots = ["baseColor", "metallicRoughness", "normalMap", "occlusion", "emissive"] as const;
const Probe = extendedMaterial(Materials, {
    name: "TextureTransformProbe",
    fragment: (ctx) => {
        "use gpu";
        const p = pbrInputFromStandardMaterial(ctx);
        const slot = ctx.parameters.baseColor.a;
        if (slot === 1) return d.vec4f(p.baseColor.xyz, 1);
        if (slot === 2) return d.vec4f(p.perceptualRoughness, p.metallic, 0, 1);
        if (slot === 3) return d.vec4f(std.add(std.mul(p.worldNormal, 0.5), d.vec3f(0.5)), 1);
        if (slot === 4) return d.vec4f(p.occlusion, 0, 0, 1);
        return d.vec4f(p.emissive, 1);
    },
});
const probePlugin = MaterialPlugin(Probe);
const subjects: { camera: number; slot: (typeof slots)[number]; channel: 0 | 1 }[] = [];
const uv = [
    [0.17, 0.31],
    [0.43, 0.19],
];
const transform = { offset: [0.41, 0.23], rotation: 0.63, scale: [0.71, 1.13] } as const;
const apps = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            probePlugin,
            {
                name: "TransformScene",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin, probePlugin],
                initialize(world) {
                    upload(world);
                    for (const [index, slot] of slots.entries())
                        for (const channel of [0, 1] as const) {
                            const x = subjects.length * 10;
                            const camera = world.create();
                            subjects.push({ camera, slot, channel });
                            world.add(camera, Transform, { translation: [x, 0, 4] });
                            world.add(camera, Camera, {
                                mode: CameraMode.Orthographic,
                                size: 1,
                                antialias: 0,
                            });
                            world.add(camera, StandardRenderer);
                            world.add(camera, Exposure, { ev100: 0 });
                            world.add(camera, Tonemapping, { method: TonemappingMethod.None });
                            attachTexture(world, camera, { width: 8, height: 8 });
                            const mesh = world.resource(Meshes).add({
                                name: `transform-${x}`,
                                vertices: new Float32Array(
                                    [
                                        [-1, -1],
                                        [1, -1],
                                        [0, 1],
                                    ].flatMap(([px, py]) => [
                                        px,
                                        py,
                                        0,
                                        uv[0][0],
                                        0,
                                        0,
                                        1,
                                        uv[0][1],
                                    ]),
                                ),
                                indices: new Uint32Array([0, 1, 2]),
                                uv1: new Float32Array([...uv[1], ...uv[1], ...uv[1]]),
                                tangent: new Float32Array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]),
                            });
                            const handle = world.resource(Probe).add(
                                StandardMaterial({
                                    baseColor: [1, 1, 1, index + 1],
                                    metallic: 1,
                                    perceptualRoughness: 1,
                                    emissive: [1, 1, 1],
                                    [`${slot}Texture`]: 0,
                                    [`${slot}Channel`]: channel,
                                    [`${slot}Transform`]: transform,
                                }),
                            );
                            const eid = world.create();
                            world.add(eid, Transform, { translation: [x, 0, 0] });
                            world.add(eid, MeshInstance, { mesh });
                            world.add(eid, MeshMaterial, handle);
                        }
                },
            },
        ],
    },
]);
// Linear data encodes a coordinate ramp; color textures store the same ramp in sRGB.
const encode = (v: number) =>
    Math.round((v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055) * 255);
function upload(world: import("../../engine").World) {
    const bindings = world.resource(imageTextures).textures;
    for (const field of ["images", "dataImages"] as const) {
        const size = 128;
        const texture = world.gpu.device.createTexture({
            size: [size, size, 1],
            format: field === "images" ? "rgba8unorm-srgb" : "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        world.own(texture);
        const pixels = new Uint8Array(size * size * 4);
        for (let y = 0; y < size; y++)
            for (let x = 0; x < size; x++) {
                const values = [(x + 0.5) / size, (y + 0.5) / size, 0.8];
                for (let c = 0; c < 3; c++)
                    pixels[(y * size + x) * 4 + c] =
                        field === "images" ? encode(values[c]) : Math.round(values[c] * 255);
                pixels[(y * size + x) * 4 + 3] = 255;
            }
        world.gpu.device.queue.writeTexture({ texture }, pixels, { bytesPerRow: size * 4 }, [
            size,
            size,
            1,
        ]);
        bindings[field === "images" ? 0 : 1][0] = texture;
    }
}
beforeAll(() => {
    const { world } = apps()[0];
    console.log("texture transform adapter", world.gpu.adapter);
    world.step(0);
    world.step(0);
}, TIMEOUT.startup);
for (const slot of slots)
    for (const channel of [0, 1] as const)
        test(`${slot} UV${channel} samples T * R * S with the specification's rotation sign`, async () => {
            const { world } = apps()[0];
            const subject = subjects.find((s) => s.slot === slot && s.channel === channel)!;
            // Column-major homogeneous matrices from KHR_texture_transform, independent of shader code.
            const c = Math.cos(transform.rotation),
                s = Math.sin(transform.rotation);
            const matrix = [
                c * transform.scale[0],
                s * transform.scale[0],
                0,
                -s * transform.scale[1],
                c * transform.scale[1],
                0,
                ...transform.offset,
                1,
            ];
            const coordinate = [0, 1].map(
                (row) =>
                    matrix[row] * uv[channel][0] +
                    matrix[3 + row] * uv[channel][1] +
                    matrix[6 + row],
            );
            let expected = [...coordinate, 0.8];
            if (slot === "metallicRoughness") expected = [coordinate[1], 0.8];
            if (slot === "occlusion") expected = [coordinate[0]];
            if (slot === "normalMap") {
                const n = [2 * coordinate[0] - 1, 2 * coordinate[1] - 1, 0.6];
                const length = Math.hypot(...n);
                expected = n.map((v) => 0.5 + (0.5 * v) / length);
            }
            const image = await captureTexture(world, subject.camera);
            const pixel = (4 * image.width + 4) * 4;
            // 128-wide ramp quantization, filtering, half-float output and display rounding.
            for (let i = 0; i < expected.length; i++)
                expect(Math.abs(image.rgba[pixel + i] - encode(expected[i]))).toBeLessThanOrEqual(
                    4,
                );
        });
