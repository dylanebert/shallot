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
import { OceanPlugin, OceanSampleGradient, oceanFragmentNormal } from "@dylanebert/shallot-ocean";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { type Check, register, type Scenario } from "../gym";

const OracleInput = d.struct({
    g0: OceanSampleGradient,
    g1: OceanSampleGradient,
    scale0: d.f32,
    scale1: d.f32,
    slope: d.vec4f,
});
const layout = tgpu.bindGroupLayout({
    input: { uniform: OracleInput },
    output: { storage: d.arrayOf(d.vec4f, 1), access: "mutable" },
});
const kernel = tgpu
    .computeFn({ workgroupSize: [1] })(() => {
        "use gpu";
        const input = layout.$.input;
        layout.$.output[0] = oceanFragmentNormal(
            input.g0,
            input.g1,
            input.scale0,
            input.scale1,
            input.slope,
        );
    })
    .$name("ocean-fragment-normal-oracle");

const fixture = {
    g0: {
        value: d.vec4f(0.1, 0.2, 0.3, 7),
        du: d.vec4f(0.04, 0.09, -0.03, 11),
        dv: d.vec4f(-0.02, 0.07, 0.05, 13),
    },
    g1: {
        value: d.vec4f(-0.2, 0.1, 0.4, 17),
        du: d.vec4f(-0.01, 0.05, 0.02, 19),
        dv: d.vec4f(0.03, -0.04, 0.06, 23),
    },
    scale0: 2.75,
    scale1: 5.5,
    slope: d.vec4f(0.35, -0.2, 29, 0.08),
};

function normalize(v: number[]): number[] {
    const length = Math.hypot(...v);
    return v.map((value) => value / length);
}

function cpu(input: typeof fixture): number[] {
    const du = [
        1 + input.g0.du.x * input.scale0 + input.g1.du.x * input.scale1,
        input.g0.du.y * input.scale0 + input.g1.du.y * input.scale1,
        input.g0.du.z * input.scale0 + input.g1.du.z * input.scale1,
    ];
    const dv = [
        input.g0.dv.x * input.scale0 + input.g1.dv.x * input.scale1,
        input.g0.dv.y * input.scale0 + input.g1.dv.y * input.scale1,
        1 + input.g0.dv.z * input.scale0 + input.g1.dv.z * input.scale1,
    ];
    const displacement = normalize([
        dv[1] * du[2] - dv[2] * du[1],
        dv[2] * du[0] - dv[0] * du[2],
        dv[0] * du[1] - dv[1] * du[0],
    ]);
    const normal = normalize([
        displacement[0] - input.slope.x,
        displacement[1],
        displacement[2] - input.slope.y,
    ]);
    return [...normal, Math.sqrt(input.slope.w)];
}

async function dispatch(inputValue: typeof fixture): Promise<Float32Array> {
    const input = Compute.root.createBuffer(OracleInput).$usage("uniform");
    input.write(inputValue);
    const output = Compute.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const group = Compute.root.createBindGroup(layout, {
        input,
        output: Compute.root.createBuffer(d.arrayOf(d.vec4f, 1), output).$usage("storage"),
    });
    const pipeline = Compute.root.createComputePipeline({ compute: kernel });
    const encoder = Compute.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pipeline.with(group).with(pass).dispatchWorkgroups(1);
    pass.end();
    Compute.device.queue.submit([encoder.finish()]);
    const result = new Float32Array((await probeBuffer(Compute.device, output)).bytes);
    output.destroy();
    input.destroy();
    return result;
}

const F32_U = 2 ** -24;
const NORMAL_OPS = 96;

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
        const expected = cpu(fixture);
        const actual = await dispatch(fixture);
        const zeroed = {
            ...fixture,
            g0: { value: d.vec4f(0), du: d.vec4f(0), dv: d.vec4f(0) },
            g1: { value: d.vec4f(0), du: d.vec4f(0), dv: d.vec4f(0) },
            slope: d.vec4f(0),
        };
        const zero = await dispatch(zeroed);
        const max = Math.max(...expected.map((value, i) => Math.abs(value - actual[i])));
        const magnitude = Math.max(1, ...expected.map(Math.abs));
        const gamma = (NORMAL_OPS * F32_U) / (1 - NORMAL_OPS * F32_U);
        const bound = gamma * magnitude;
        const witness = Math.max(...expected.map((value, i) => Math.abs(value - zero[i])));
        checks = [
            {
                name: "device executes the shipped fragment-normal kernel against the CPU closed form",
                pass: max <= bound,
                detail: `max=${max} bound=${bound} (u=${F32_U}, ops=${NORMAL_OPS}, magnitude=${magnitude})`,
            },
            {
                name: "zeroed displacement/slope payload reds the same comparison",
                pass: witness > bound,
                detail: `zeroedDeviation=${witness} sharedBound=${bound}`,
            },
        ];
        return built;
    },
    async assert() {
        return checks;
    },
};
register(scenario);
