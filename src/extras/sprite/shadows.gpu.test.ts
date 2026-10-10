import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, DirectionalLight } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import {
    DirectionalLightShadowMap,
    StandardRenderer,
    sunShadowView,
} from "../../standard/rendering";
import { Sprite, SpriteBillboard, SpriteFill, SpritePlugin } from "./index";

setDefaultTimeout(CEILING.gpu);
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [SpritePlugin] }]);
const SHADOW_SIDE = 256;

const pixelShader = `
@group(0) @binding(0) var shadowMap: texture_depth_2d;
@group(0) @binding(1) var<storage, read_write> depths: array<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(shadowMap);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    depths[gid.y * dims.x + gid.x] = textureLoad(shadowMap, vec2i(gid.xy), 0);
}`;

test("clip Sprite leaves its alpha hole out of the directional shadow map", async () => {
    const { world } = subjects()[0];
    world.resource(DirectionalLightShadowMap).size = SHADOW_SIDE;
    const device = world.gpu.device;
    // Dawn's Node harness has no createImageBitmap. Bind a known opaque texel directly so the witness
    // measures clip discard in the shadow pass rather than host image decoding.
    const atlas = device.createTexture({
        size: [1, 1, 1],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(atlas);
    device.queue.writeTexture(
        { texture: atlas },
        new Uint8Array([255, 255, 255, 255]),
        { bytesPerRow: 4, rowsPerImage: 1 },
        { width: 1, height: 1, depthOrArrayLayers: 1 },
    );
    world.gpu.textures.set("spriteAtlas", atlas);

    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 32, height: 32 });

    const receiver = world.create();
    world.add(receiver, Transform, {
        translation: [0, 0, -2, 0],
        scale: [3, 3, 0.1, 0],
    });
    world.add(receiver, MeshInstance);
    const sun = world.create();
    const sunRotation = lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5);
    world.add(sun, Transform, {
        rotation: [sunRotation.x, sunRotation.y, sunRotation.z, sunRotation.w],
    });
    world.add(sun, DirectionalLight, { shadowMapsEnabled: 1, numCascades: 1 });

    const depthBuffer = device.createBuffer({
        size: SHADOW_SIDE * SHADOW_SIDE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    world.own(depthBuffer);
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "depth" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
    });
    const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: device.createShaderModule({ code: pixelShader }), entryPoint: "main" },
    });
    async function depthPixels(): Promise<Float32Array> {
        const view = sunShadowView(world);
        expect(view).not.toBeNull();
        const group = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: view! },
                { binding: 1, resource: { buffer: depthBuffer } },
            ],
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(SHADOW_SIDE / 8, SHADOW_SIDE / 8);
        pass.end();
        device.queue.submit([encoder.finish()]);
        const snapshot = await probeBuffer(world, depthBuffer, {
            size: SHADOW_SIDE * SHADOW_SIDE * 4,
        });
        return new Float32Array(snapshot.bytes);
    }

    world.step(0);
    world.step(0);
    const before = await depthPixels();

    const sprite = world.create();
    world.add(sprite, Transform);
    world.add(sprite, Sprite, {
        size: [2, 2],
        fill: 0.5,
        fillMode: SpriteFill.Horizontal,
        billboard: SpriteBillboard.World,
    });
    world.step(0);
    world.step(0);
    const after = await depthPixels();
    let leftChanged = 0;
    let rightChanged = 0;
    for (let y = 0; y < SHADOW_SIDE; y++) {
        for (let x = 0; x < SHADOW_SIDE; x++) {
            const index = y * SHADOW_SIDE + x;
            if (Math.abs(after[index]! - before[index]!) < 1e-5) continue;
            if (x < SHADOW_SIDE / 2) leftChanged++;
            else rightChanged++;
        }
    }
    console.log("clip-sprite shadow depth changes", { leftChanged, rightChanged });
    expect(Math.max(leftChanged, rightChanged)).toBeGreaterThan(0);
    expect(Math.min(leftChanged, rightChanged)).toBeLessThan(
        Math.max(leftChanged, rightChanged) / 4,
    );
});
