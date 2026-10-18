import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    DirectionalLight,
    Exposure,
    VolumetricLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { probeTexture } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import {
    Fog,
    FogPlugin,
    fogReadBindGroup,
    fogReadLayout,
    NoFog,
    readFog,
    readFogInScatter,
} from ".";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);

test("local light is dark behind a shadow despite bright haze on its ray, exposed once, and zero without fog", async () => {
    const { world } = subjects()[0]!;
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera, { antialias: 0, near: 0.1 });
    world.add(camera, StandardRenderer);
    world.add(camera, AmbientLight, { brightness: 0 });
    world.add(camera, Exposure, { ev100: 0 });
    attachTexture(world, camera, { width: 48, height: 48 });
    const fog = world.create();
    world.add(fog, Fog, {
        color: 0,
        density: 0.1,
        heightFalloff: 0,
        scatteringAsymmetry: 0,
        viewDistance: 20,
        // Keep the endpoint guard from masking a reversed ray before the near plane.
        maxDistance: 1_000_000,
    });
    const sun = world.create();
    world.add(sun, Transform, { rotation: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2] });
    world.add(sun, DirectionalLight, {
        illuminance: 1000,
        shadowMapsEnabled: 1,
        maximumDistance: 30,
    });
    world.add(sun, VolumetricLight);
    const roof = world.create();
    world.add(roof, Transform, { translation: [0, 2, -10], scale: [10, 0.2, 8] });
    world.add(roof, MeshInstance);
    world.tick();
    const device = world.gpu.device;
    const module = device.createShaderModule({
        code: tgpu.resolve({
            externals: { readFog, readFogInScatter },
            template: `
        @vertex fn fullscreenVertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            let x = f32((i << 1u) & 2u); let y = f32(i & 2u);
            return vec4f(x * 2 - 1, y * 2 - 1, 0, 1);
        }
        @fragment fn sample(@builtin(position) p: vec4f) -> @location(0) vec4f {
            if (p.x < 1) { return vec4f(readFogInScatter(vec3f(0,0,-3)), 1); }
            if (p.x < 2) { return vec4f(readFogInScatter(vec3f(0,0,-11)), 1); }
            if (p.x < 3) { return readFog(vec3f(0,0,-11)); }
            if (p.x < 4) { return vec4f(readFogInScatter(vec3f(0,0,-0.05)), 1); }
            return vec4f(readFogInScatter(vec3f(0,0,-0.1)), 1);
        }
        `,
        }),
    });
    const empty = device.createBindGroupLayout({ entries: [] });
    const pipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [empty, empty, empty, world.gpu.root.unwrap(fogReadLayout)],
        }),
        vertex: { module, entryPoint: "fullscreenVertex" },
        fragment: { module, entryPoint: "sample", targets: [{ format: "rgba32float" }] },
    });
    const output = device.createTexture({
        size: [5, 1],
        format: "rgba32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    world.own(output);
    const sample = async () => {
        for (let i = 0; i < 64; i++) world.step(0);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: output.createView(), loadOp: "clear", storeOp: "store" }],
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(3, fogReadBindGroup(world, camera));
        pass.draw(3);
        pass.end();
        device.queue.submit([encoder.finish()]);
        return new Float32Array((await probeTexture(world, output)).bytes);
    };
    const lit = await sample();
    expect(lit[8]!).toBeGreaterThan(0.1);
    expect(lit[0]!).toBeGreaterThan(1);
    expect(lit[4]!).toBeLessThan(lit[0]! * 0.02);
    world.storage(AmbientLight).brightness.set(camera, 100);
    const boundary = await sample();
    expect({
        beforeNear: Array.from(boundary.slice(12, 15)),
        atNear: Array.from(boundary.slice(16, 19)),
    }).toEqual({ beforeNear: [0, 0, 0], atNear: [0, 0, 0] });
    world.storage(AmbientLight).brightness.set(camera, 0);
    world.storage(Exposure).ev100.set(camera, 1);
    const half = await sample();
    expect(half[0]! / lit[0]!).toBeCloseTo(0.5, 2);
    world.storage(Fog).density.set(fog, 0);
    expect((await sample())[0]).toBe(0);
    world.storage(Fog).density.set(fog, 0.1);
    world.add(camera, NoFog);
    expect((await sample())[0]).toBe(0);
    world.remove(camera, NoFog);
    world.remove(fog, Fog);
    expect((await sample())[0]).toBe(0);
});
