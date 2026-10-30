import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
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
    AlphaMode,
    applyPbrLighting,
    extendedMaterial,
    type MaterialHandle,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    pbrInputFromStandardMaterial,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";
import { brdf, Pbr } from "./shade";

setDefaultTimeout(TIMEOUT.gpu);
const light = std.normalize(d.vec3f(0.7, 0.3, 1));
const view = d.vec3f(0, 0, 1);
const Surface = extendedMaterial(Materials, {
    name: "TextureShading",
    fragment: (ctx) => {
        "use gpu";
        const s = pbrInputFromStandardMaterial(ctx);
        // Fix the view direction so the CPU formula does not depend on pixel-center placement.
        s.view = view;
        return applyPbrLighting(s);
    },
});
const NormalProbe = extendedMaterial(Materials, {
    name: "BackNormalProbe",
    fragment: (ctx) => {
        "use gpu";
        const s = pbrInputFromStandardMaterial(ctx);
        return d.vec4f(std.add(std.mul(s.worldNormal, 0.5), d.vec3f(0.5)), 1);
    },
});
const modes = [
    "normal",
    "metallic-roughness",
    "occlusion",
    "emissive",
    "vertex color",
    "tangent interpolation",
    "back normal",
] as const;
const samples = [
    [64, 180, 224],
    [204, 80, 192],
];
const cameras: { mode: (typeof modes)[number]; channel: 0 | 1; sign: number; camera: number }[] =
    [];
// Unit and orthogonal at vertex 2; interpolation must retain its length and normal component.
const tiltedTangent = [0.15, 0.2 * Math.sqrt(0.9775), Math.sqrt(0.96 * 0.9775)] as const;
const alphaSubjects: { camera: number; channel: 0 | 1; handle: MaterialHandle }[] = [];
const SurfacePlugin = MaterialPlugin(Surface);
const NormalProbePlugin = MaterialPlugin(NormalProbe);
const apps = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            SurfacePlugin,
            NormalProbePlugin,
            {
                name: "TextureShadingScene",
                dependencies: [
                    StandardRenderingPlugin,
                    MeshRenderPlugin,
                    SurfacePlugin,
                    NormalProbePlugin,
                ],
                initialize(world) {
                    const sun = world.create();
                    world.add(sun, Transform, {
                        rotation: lookAtRotation(0, 0, 0, -light.x, -light.y, -light.z),
                    });
                    world.add(sun, DirectionalLight, { illuminance: 1.2, shadowMapsEnabled: 0 });
                    for (const mode of modes)
                        for (const channel of [0, 1] as const)
                            for (const sign of mode === "normal" ? [1, -1] : [1]) {
                                const origin = cameras.length * 10;
                                const camera = world.create();
                                cameras.push({ mode, channel, sign, camera });
                                world.add(camera, Transform, { translation: [origin, 0, 4] });
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
                                const mesh = world.resource(Meshes).add({
                                    name: `surface-${origin}`,
                                    vertices: new Float32Array(
                                        [
                                            [-1, -1],
                                            [1, -1],
                                            [0, 1],
                                        ].flatMap(([x, y], vertex) => [
                                            x,
                                            y,
                                            0,
                                            0.25,
                                            ...(mode === "normal" ||
                                            mode === "tangent interpolation"
                                                ? [
                                                      [0, 0, 1],
                                                      [0, Math.sqrt(0.96), 0.2],
                                                      [0, -Math.sqrt(0.96), 0.2],
                                                  ][vertex]
                                                : [0, 0, 1]),
                                            0.5,
                                        ]),
                                    ),
                                    indices: new Uint32Array(
                                        mode === "back normal" ? [0, 2, 1] : [0, 1, 2],
                                    ),
                                    uv1: new Float32Array([0.75, 0.5, 0.75, 0.5, 0.75, 0.5]),
                                    tangent: new Float32Array([
                                        1,
                                        0,
                                        0,
                                        sign,
                                        mode === "tangent interpolation" ? -1 : 1,
                                        0,
                                        0,
                                        sign,
                                        ...(mode === "tangent interpolation"
                                            ? tiltedTangent
                                            : [1, 0, 0]),
                                        sign,
                                    ]),
                                    ...(mode === "vertex color"
                                        ? {
                                              color: new Float32Array([
                                                  0.6, 0.8, 0.4, 0.7, 0.6, 0.8, 0.4, 0.7, 0.6, 0.8,
                                                  0.4, 0.7,
                                              ]),
                                          }
                                        : {}),
                                });
                                const handle = world
                                    .resource(mode === "back normal" ? NormalProbe : Surface)
                                    .add(
                                        StandardMaterial({
                                            baseColor: [0.3, 0.4, 0.2, 1],
                                            metallic: mode === "tangent interpolation" ? 0 : 0.8,
                                            perceptualRoughness: 0.9,
                                            doubleSided: mode === "back normal",
                                            ...(mode === "normal" ||
                                            mode === "tangent interpolation" ||
                                            mode === "back normal"
                                                ? {
                                                      normalMapTexture: 0,
                                                      normalMapChannel: channel,
                                                      normalMapScale: 0.5,
                                                  }
                                                : {}),
                                            ...(mode === "metallic-roughness"
                                                ? {
                                                      metallicRoughnessTexture: 0,
                                                      metallicRoughnessChannel: channel,
                                                  }
                                                : {}),
                                            ...(mode === "occlusion"
                                                ? {
                                                      occlusionTexture: 0,
                                                      occlusionChannel: channel,
                                                      occlusionStrength: 0.6,
                                                  }
                                                : {}),
                                            ...(mode === "emissive"
                                                ? {
                                                      emissiveTexture: 0,
                                                      emissiveChannel: channel,
                                                      emissive: [0.2, 0.3, 0.4],
                                                  }
                                                : {}),
                                            ...(mode === "vertex color"
                                                ? { baseColorTexture: 0, baseColorChannel: channel }
                                                : {}),
                                        }),
                                        {
                                            cullMode: mode === "back normal" ? "none" : "back",
                                            ...(mode === "vertex color"
                                                ? { alphaMode: AlphaMode.Mask(0.5) }
                                                : {}),
                                        },
                                    );
                                if (mode === "vertex color")
                                    alphaSubjects.push({ camera, channel, handle });
                                const eid = world.create();
                                world.add(eid, Transform, { translation: [origin, 0, 0] });
                                world.add(eid, MeshInstance, { mesh });
                                world.add(eid, MeshMaterial, handle);
                            }
                },
            },
        ],
    },
]);
function upload(world: import("../../engine").World) {
    const bindings = world.resource(ImageBindings);
    for (const [field, format] of [
        ["images", "rgba8unorm-srgb"],
        ["dataImages", "rgba8unorm"],
    ] as const) {
        const texture = world.gpu.device.createTexture({
            size: [2, 1, 1],
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        world.own(texture);
        world.gpu.device.queue.writeTexture(
            { texture },
            new Uint8Array([...samples[0], 255, ...samples[1], 255]),
            { bytesPerRow: 8 },
            [2, 1, 1],
        );
        bindings[field] = texture;
    }
}
beforeAll(() => {
    const { world } = apps()[0];
    upload(world);
    console.log("texture shading adapter", world.gpu.adapter);
    world.step(0);
    world.step(0);
}, TIMEOUT.startup);
const linear = (byte: number) => {
    const v = byte / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};
const display = (v: number) =>
    Math.round(Math.min(1, v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055) * 255);
// Subjects are created during initialize, so enumerate their identical authoring table here.
for (const mode of modes)
    for (const channel of [0, 1] as const)
        for (const sign of mode === "normal" ? [1, -1] : [1])
            test(`${mode} UV${channel} handedness ${sign}: ${mode === "back normal" ? "double-sided reverses the complete mapped normal" : "shaded texel equals CPU shade.ts on independently specified channels"}`, async () => {
                const subject = cameras.find(
                    (c) => c.mode === mode && c.channel === channel && c.sign === sign,
                )!;
                const sample = samples[channel];
                const base = [0.3, 0.4, 0.2].map((v, i) =>
                    mode === "vertex color" ? v * [0.6, 0.8, 0.4][i] * linear(sample[i]) : v,
                );
                const metallic =
                    mode === "tangent interpolation"
                        ? 0
                        : 0.8 * (mode === "metallic-roughness" ? sample[2] / 255 : 1);
                const roughness = 0.9 * (mode === "metallic-roughness" ? sample[1] / 255 : 1);
                const occlusion = mode === "occlusion" ? 1 + 0.6 * (sample[0] / 255 - 1) : 1;
                // Pixel (8,8) is world (1/16,-1/16): barycentrics (15,19,30)/64.
                // Interpolate the authored unit normals without normalizing the basis.
                const ny = (Math.sqrt(0.96) * (19 - 30)) / 64;
                const nz = (15 + 0.2 * (19 + 30)) / 64;
                const mx = ((2 * sample[0]) / 255 - 1) * 0.5;
                const my = ((2 * sample[1]) / 255 - 1) * 0.5 * sign;
                const mz = (2 * sample[2]) / 255 - 1;
                const tx =
                    mode === "tangent interpolation" ? (15 - 19 + 30 * tiltedTangent[0]) / 64 : 1;
                const ty = mode === "tangent interpolation" ? (30 * tiltedTangent[1]) / 64 : 0;
                const tz = mode === "tangent interpolation" ? (30 * tiltedTangent[2]) / 64 : 0;
                const normal =
                    mode === "normal" || mode === "tangent interpolation"
                        ? std.normalize(
                              d.vec3f(
                                  tx * mx + (ny * tz - nz * ty) * my,
                                  ty * mx + tx * nz * my + ny * mz,
                                  tz * mx - tx * ny * my + nz * mz,
                              ),
                          )
                        : d.vec3f(0, 0, 1);
                const direct = brdf(
                    Pbr({
                        albedo: d.vec3f(...(base as [number, number, number])),
                        metallic,
                        roughness,
                        occlusion,
                        dielectric: 0,
                    }),
                    normal,
                    view,
                    light,
                );
                const expected =
                    mode === "back normal"
                        ? [...std.normalize(d.vec3f(mx, my, mz))].map((v) => display(0.5 - 0.5 * v))
                        : [...direct].map((v, i) =>
                              display(
                                  v +
                                      base[i] * 0.2 * occlusion +
                                      (mode === "emissive"
                                          ? [0.2, 0.3, 0.4][i] * linear(sample[i])
                                          : 0),
                              ),
                          );
                const image = await captureTexture(apps()[0].world, subject.camera);
                const pixel = [...image.rgba.subarray((8 * 16 + 8) * 4, (8 * 16 + 8) * 4 + 3)];
                for (let i = 0; i < 3; i++)
                    expect(Math.abs(pixel[i] - expected[i])).toBeLessThanOrEqual(2);
            });

for (const channel of [0, 1] as const)
    test(`vertex color UV${channel}: alpha multiplies base alpha before mask coverage`, async () => {
        const { world } = apps()[0];
        const { camera, handle } = alphaSubjects.find((s) => s.channel === channel)!;
        const center = (rgba: Uint8ClampedArray) => [
            ...rgba.subarray((8 * 16 + 8) * 4, (8 * 16 + 8) * 4 + 4),
        ];
        const before = await captureTexture(world, camera);
        expect(center(before.rgba)).not.toEqual([...before.rgba.subarray(0, 4)]);
        // Authored color alpha is 0.7: 0.6 * 0.7 = 0.42, below the 0.5 cutoff.
        // Ignoring color alpha leaves 0.6, which would retain the shaded pixel.
        world.resource(Surface).update(handle, { baseColor: [0.3, 0.4, 0.2, 0.6] });
        world.step(0);
        world.step(0);
        const after = await captureTexture(world, camera);
        expect(center(after.rgba)).toEqual([...after.rgba.subarray(0, 4)]);
    });
