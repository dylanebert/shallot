import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { type Plugin, probeTexture } from "../../engine";
import { Transform } from "../transform";
import {
    attachTexture,
    Camera,
    CorePipelinePlugin,
    detachView,
    RenderContext,
    RenderPhases,
} from "./index";
import { viewResourcesKey } from "./view";

setDefaultTimeout(TIMEOUT.gpu);
const Scene: Plugin = {
    name: "ScenePrecisionRamp",
    dependencies: [CorePipelinePlugin],
    initialize(world) {
        const module = world.gpu.device.createShaderModule({
            code: `
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.5, 1.0);
}
@fragment fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4f {
    return vec4f(vec3f(0.1 + p.x / 8192.0 * 0.6), 1.0);
}`,
        });
        const pipeline = world.gpu.device.createRenderPipeline({
            layout: "auto",
            vertex: { module, entryPoint: "vertex" },
            fragment: {
                module,
                entryPoint: "fragment",
                targets: [{ format: world.resource(RenderContext).format }],
            },
            depthStencil: {
                format: "depth32float",
                depthWriteEnabled: false,
                depthCompare: "always",
            },
        });
        world.resource(RenderPhases).push({
            opaque(_world, _eid, _view, pass) {
                pass.setPipeline(pipeline);
                pass.draw(3);
            },
        });
    },
};
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [Scene] }]);

function half(bits: number) {
    const exponent = (bits >>> 10) & 31;
    const mantissa = bits & 1023;
    return (
        (bits & 32768 ? -1 : 1) *
        (exponent ? (1 + mantissa / 1024) * 2 ** (exponent - 15) : mantissa * 2 ** -24)
    );
}
function displayCode(radiance: number) {
    const linear = radiance / (1 + radiance);
    return 255 * (1.055 * linear ** (1 / 2.4) - 0.055);
}

test("scene HDR ramp has no adjacent step of 0.1 display code after Reinhard and sRGB", async () => {
    const { world } = subjects()[0];
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Camera);
    world.storage(Camera).antialias.set(eid, 0);
    attachTexture(world, eid, { width: 8192, height: 1 });
    try {
        world.step(0);
        const scene = world.resource(viewResourcesKey).offscreen.get(eid)!.texture;
        const result = await probeTexture(world, scene);
        const data = new DataView(result.bytes);
        let largest = 0;
        for (let channel = 0; channel < 3; channel++) {
            let previous = 0;
            for (let x = 0; x < 8192; x++) {
                const bits =
                    result.format === "rgba16float"
                        ? data.getUint16(x * 8 + channel * 2, true)
                        : channel === 2
                          ? ((data.getUint32(x * 4, true) >>> 22) & 1023) << 5
                          : ((data.getUint32(x * 4, true) >>> (channel * 11)) & 2047) << 4;
                const code = displayCode(half(bits));
                if (x) largest = Math.max(largest, code - previous);
                previous = code;
            }
        }
        console.info(`scene ramp ${result.format}: largest adjacent step ${largest} display codes`);
        expect(largest).toBeLessThan(0.1);
    } finally {
        detachView(world, eid);
        world.destroy(eid);
    }
});
