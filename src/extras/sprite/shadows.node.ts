import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Billboard,
    Camera,
    captureTexture,
    DirectionalLight,
} from "../../core/rendering";
import { ImageBindings } from "../../core/rendering/image";
import { attachView, detachView, Views } from "../../core/rendering/view";
import { Transform } from "../../core/transform";
import { lookAtRotation, precompile } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import {
    DirectionalLightShadowMap,
    StandardRenderer,
    sunShadowView,
} from "../../standard/rendering";
import { Sprite, SpriteFill, SpritePlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
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

function shadowSubject(depthOnly = false) {
    return gpuScene(import.meta.path, { defaults: false, plugins: [SpritePlugin] }, (world) => {
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

        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        let sibling = -1;
        if (depthOnly) {
            attachView(world, camera);
            sibling = world.create();
            world.add(sibling, Transform, { translation: [0, 0, 5, 0] });
            world.add(sibling, Camera);
            world.add(sibling, StandardRenderer);
            attachTexture(world, sibling, { width: 32, height: 32 });
        } else {
            attachTexture(world, camera, { width: 32, height: 32 });
        }

        const receiver = world.create();
        world.add(receiver, Transform, {
            translation: [0, 0, -2, 0],
            scale: [3, 3, 0.1, 0],
        });
        world.add(receiver, MeshInstance);
        const sun = world.create();
        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
        world.add(sun, DirectionalLight, { shadowMapsEnabled: 1, numCascades: 1 });

        const depthBuffer = device.createBuffer({
            size: SHADOW_SIDE * SHADOW_SIDE * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        world.own(depthBuffer);
        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "depth" },
                },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
        const pipeline = device.createComputePipeline({
            label: "sprite-shadow-pixels",
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: pixelShader }),
                entryPoint: "main",
            },
        });
        precompile(world, "sprite-shadow-pixels", () => [pipeline]);
        return {
            world,
            device,
            atlas,
            depthBuffer,
            bindGroupLayout,
            pipeline,
            camera,
            sun,
            sibling,
        };
    });
}
const subject = shadowSubject();
const depthOnlySubject = shadowSubject(true);
test("clip Sprite leaves its alpha hole out of the directional shadow map", async () => {
    const {
        scene: { world, atlas },
    } = subject();
    world.resource(ImageBindings).images = atlas;
    const depthPixels = () => readDepth(subject().scene);

    world.step(0);
    world.step(0);
    const before = await depthPixels();

    const sprite = world.create();
    world.add(sprite, Transform);
    world.add(sprite, Sprite, {
        size: [2, 2],
        fillAmount: 0.5,
        fillMethod: SpriteFill.Horizontal,
        billboard: Billboard.World,
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
    world.remove(sprite, Sprite);
});

async function readDepth({
    world,
    device,
    depthBuffer,
    bindGroupLayout,
    pipeline,
}: ReturnType<typeof subject>["scene"]): Promise<Float32Array> {
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

test("a Screen-billboard Sprite's sun shadow faces the camera the shadows are fit to, whichever camera holds view slot 0", async () => {
    const { scene } = subject();
    const { world, camera, sun } = scene;
    const other = world.create();
    world.add(other, Transform, {
        translation: [5, 0, 0, 0],
        rotation: lookAtRotation(5, 0, 0, 0, 0, 0),
    });
    world.add(other, Camera);
    world.add(other, StandardRenderer);
    const views = world.resource(Views);
    detachView(world, camera);
    const target = attachTexture(world, camera, { width: 32, height: 32 });
    attachTexture(world, other, target);
    world.storage(Camera).order.set(other, 1);
    world.storage(DirectionalLight).maximumDistance.set(sun, 8);
    world.step(0);
    expect(views.get(camera)!.slot).toBe(0);
    const without = await readDepth(scene);
    const sprite = world.create();
    world.add(sprite, Transform);
    world.add(sprite, Sprite, { size: [2, 2], billboard: Billboard.Screen });
    world.step(0);
    const beforeSwap = await readDepth(scene);
    expect(beforeSwap.some((depth, i) => depth !== without[i])).toBe(true);
    world.storage(Camera).order.set(camera, 2);
    world.step(0);
    expect(views.get(other)!.slot).toBe(0);
    expect(views.get(camera)!.slot).toBeGreaterThan(0);
    const withOther = await readDepth(scene);
    world.remove(other, StandardRenderer);
    world.remove(other, Camera);
    world.step(0);
    const alone = await readDepth(scene);
    let changed = 0;
    let minimum = Infinity;
    let maximum = -Infinity;
    for (let i = 0; i < alone.length; i++) {
        if (alone[i] !== withOther[i]) changed++;
        minimum = Math.min(minimum, alone[i]!, withOther[i]!);
        maximum = Math.max(maximum, alone[i]!, withOther[i]!);
    }
    expect(minimum).toBeGreaterThanOrEqual(0);
    expect(maximum).toBeLessThanOrEqual(1);
    expect(changed).toBe(0);
    expect(new Uint8Array(alone.buffer)).toEqual(new Uint8Array(withOther.buffer));
});

test("a depth-only main camera with a presenting sibling records shadows and presents its first frame", async () => {
    const { scene } = depthOnlySubject();
    const { world, camera, sibling, sun, atlas } = scene;
    world.resource(ImageBindings).images = atlas;
    world.storage(DirectionalLight).maximumDistance.set(sun, 8);
    const sprite = world.create();
    world.add(sprite, Transform);
    world.add(sprite, Sprite, { size: [2, 2], billboard: Billboard.Screen });
    expect(() => world.step(0)).not.toThrow();
    const views = world.resource(Views);
    expect(views.get(sibling)!.slot).toBe(0);
    expect(views.get(camera)!.slot).toBeGreaterThan(0);
    const depths = await readDepth(scene);
    let minimum = Infinity;
    let maximum = -Infinity;
    for (const depth of depths) {
        minimum = Math.min(minimum, depth);
        maximum = Math.max(maximum, depth);
    }
    expect(minimum).toBeGreaterThanOrEqual(0);
    expect(maximum).toBeLessThanOrEqual(1);
    expect(minimum).toBeLessThan(1);
    const { rgba } = await captureTexture(world, sibling);
    expect(rgba.length).toBe(32 * 32 * 4);
    expect(rgba.some((value, index) => index % 4 !== 3 && value > 0)).toBe(true);
});
