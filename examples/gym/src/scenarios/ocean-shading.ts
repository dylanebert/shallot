import {
    Compute,
    PartPlugin,
    probeBuffer,
    RenderPlugin,
    run,
    SearPlugin,
    SlabPlugin,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import {
    OceanPlugin,
    surfaceCatmullRom1D,
    surfaceCatmullRomDerivative1D,
} from "@dylanebert/shallot-ocean";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { type Check, register, type Scenario } from "../gym";

const layout = tgpu.bindGroupLayout({
    controls: { storage: d.arrayOf(d.vec4f, 5), access: "readonly" },
    output: { storage: d.arrayOf(d.vec4f, 1), access: "mutable" },
});

const kernel = tgpu
    .computeFn({ workgroupSize: [1] })(() => {
        "use gpu";
        const derivative = surfaceCatmullRomDerivative1D(
            layout.$.controls[0],
            layout.$.controls[1],
            layout.$.controls[2],
            layout.$.controls[3],
            0.37,
        );
        const value = surfaceCatmullRom1D(
            layout.$.controls[0],
            layout.$.controls[1],
            layout.$.controls[2],
            layout.$.controls[3],
            0.37,
        );
        const slope = layout.$.controls[4];
        const normal = std.normalize(d.vec3f(-derivative.y - slope.x, 1, -derivative.z - slope.y));
        layout.$.output[0] = d.vec4f(normal, value.y + slope.w);
    })
    .$name("ocean-shading-oracle");

function cpu(controls: Float32Array): number[] {
    const t = 0.37;
    const deriv = (c: number) => {
        const p0 = controls[c];
        const p1 = controls[4 + c];
        const p2 = controls[8 + c];
        const p3 = controls[12 + c];
        const a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
        const b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
        return -0.5 * p0 + 0.5 * p2 + t * (2 * b + 3 * t * a);
    };
    const value = (c: number) => {
        const p0 = controls[c];
        const p1 = controls[4 + c];
        const p2 = controls[8 + c];
        const p3 = controls[12 + c];
        const a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
        const b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
        const cc = -0.5 * p0 + 0.5 * p2;
        return p1 + t * (cc + t * (b + t * a));
    };
    const x = -deriv(1) - controls[16];
    const z = -deriv(2) - controls[17];
    const length = Math.hypot(x, 1, z);
    return [x / length, 1 / length, z / length, value(1) + controls[19]];
}

async function dispatch(values: Float32Array): Promise<Float32Array> {
    const input = Compute.device.createBuffer({
        size: values.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const output = Compute.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    Compute.device.queue.writeBuffer(input, 0, values);
    const group = Compute.root.createBindGroup(layout, {
        controls: Compute.root.createBuffer(d.arrayOf(d.vec4f, 5), input).$usage("storage"),
        output: Compute.root.createBuffer(d.arrayOf(d.vec4f, 1), output).$usage("storage"),
    });
    const pipeline = Compute.root.createComputePipeline({ compute: kernel });
    const encoder = Compute.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pipeline.with(group).with(pass).dispatchWorkgroups(1);
    pass.end();
    Compute.device.queue.submit([encoder.finish()]);
    const result = new Float32Array((await probeBuffer(Compute.device, output)).bytes);
    input.destroy();
    output.destroy();
    return result;
}

let checks: Check[] = [];
const scenario: Scenario = {
    name: "ocean-shading",
    noRender: true,
    async build() {
        const built = await run({
            defaults: false,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                TransformsPlugin,
                RenderPlugin,
                PartPlugin,
                SearPlugin,
                OceanPlugin,
            ],
        });
        built.state.pause();
        const values = new Float32Array([
            0.1, 0.2, -0.3, 0, 0.4, 0.7, 0.2, 0, -0.2, 1.1, 0.6, 0, 0.3, -0.1, 0.9, 0, 0.35, -0.2,
            0, 0.08,
        ]);
        const expected = cpu(values);
        const actual = await dispatch(values);
        const zero = await dispatch(new Float32Array(20));
        const max = Math.max(...expected.map((value, i) => Math.abs(value - actual[i])));
        const witness = Math.max(...expected.map((value, i) => Math.abs(value - zero[i])));
        checks = [
            {
                name: "device shading normal agrees with the CPU closed form",
                pass: max <= 2 ** -20,
                detail: `max=${max}`,
            },
            {
                name: "zeroed displacement/slope payload reds the same comparison",
                pass: witness > 2 ** -20,
                detail: `zeroedDeviation=${witness}`,
            },
        ];
        return built;
    },
    async assert() {
        return checks;
    },
};
register(scenario);
