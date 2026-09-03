import { Compute, probeBuffer, type State } from "@dylanebert/shallot";
import { Surfaces } from "@dylanebert/shallot/sear/core";
import { Xform } from "@dylanebert/shallot/utils/core";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { oceanEstimateDisplacement, oceanFragmentNormal, oceanSurfaceLayout } from "../ocean/index";
import { DuskSkyGpu } from "../sky";

interface Check {
    name: string;
    pass: boolean;
    detail?: string;
}

const ProbeParams = d
    .struct({ world: d.vec2f, slopeTexel: d.vec2i })
    .$name("OceanShadingProbeParams");
const probeParamsLayout = tgpu
    .bindGroupLayout({ params: { uniform: ProbeParams } })
    .$idx(0)
    .$name("ocean-shading-probe-params");
const probeOutputLayout = tgpu
    .bindGroupLayout({ output: { storage: d.arrayOf(d.vec4f, 1), access: "mutable" } })
    .$idx(1)
    .$name("ocean-shading-probe-output");
const kernel = tgpu
    .computeFn({ workgroupSize: [1] })(() => {
        "use gpu";
        const estimate = oceanEstimateDisplacement(probeParamsLayout.$.params.world);
        const slope = std.textureLoad(
            oceanSurfaceLayout.$.slope0,
            probeParamsLayout.$.params.slopeTexel,
            0,
        );
        probeOutputLayout.$.output[0] = oceanFragmentNormal(
            estimate.g0,
            estimate.g1,
            estimate.scale0,
            estimate.scale1,
            slope,
        );
    })
    .$name("ocean-shipped-estimator-oracle");

const N = 8;
const WORLD = d.vec2f(0, 0);
const SLOPE = [0.3125, -0.1875, 0, 0.078125] as const;

function texture(values: Float32Array, label: string, mipLevelCount = 1): GPUTexture {
    const result = Compute.device.createTexture({
        label,
        size: [N, N],
        mipLevelCount,
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    Compute.device.queue.writeTexture(
        { texture: result },
        new Float16Array(values),
        { bytesPerRow: N * 8, rowsPerImage: N },
        [N, N],
    );
    return result;
}

function displacement(coefficients: readonly number[]): Float32Array {
    const out = new Float32Array(N * N * 4);
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const i = (y * N + x) * 4;
            out[i] = coefficients[0] * x + coefficients[1] * y;
            out[i + 1] = coefficients[2] * x + coefficients[3] * y;
            out[i + 2] = coefficients[4] * x + coefficients[5] * y;
            out[i + 3] = 31 + x + N * y;
        }
    }
    return out;
}

const COEFFICIENTS = [
    [0.015625, -0.0078125, 0.03125, 0.01171875, -0.01953125, 0.0234375],
    [-0.01171875, 0.01953125, 0.0078125, -0.015625, 0.02734375, -0.00390625],
] as const;

function normalize(v: number[]): number[] {
    const length = Math.hypot(...v);
    return v.map((value) => value / length);
}

function expected(zeroed: boolean): number[] {
    if (zeroed) return [0, 1, 0, 0];
    const scales = [N / 80, N / 31];
    const du = [1, 0, 0];
    const dv = [0, 0, 1];
    for (let cascade = 0; cascade < 2; cascade++) {
        const c = COEFFICIENTS[cascade];
        du[0] += c[0] * scales[cascade];
        du[1] += c[2] * scales[cascade];
        du[2] += c[4] * scales[cascade];
        dv[0] += c[1] * scales[cascade];
        dv[1] += c[3] * scales[cascade];
        dv[2] += c[5] * scales[cascade];
    }
    const displacementNormal = normalize([
        dv[1] * du[2] - dv[2] * du[1],
        dv[2] * du[0] - dv[0] * du[2],
        dv[0] * du[1] - dv[1] * du[0],
    ]);
    const normal = normalize([
        displacementNormal[0] - SLOPE[0],
        displacementNormal[1],
        displacementNormal[2] - SLOPE[1],
    ]);
    return [...normal, Math.sqrt(SLOPE[3])];
}

async function dispatch(zeroed: boolean): Promise<Float32Array> {
    const fields = zeroed
        ? [new Float32Array(N * N * 4), new Float32Array(N * N * 4)]
        : COEFFICIENTS.map(displacement);
    const displace0 = texture(fields[0], `ocean-oracle-displace0-${zeroed}`);
    const displace1 = texture(fields[1], `ocean-oracle-displace1-${zeroed}`);
    const slopeValues = new Float32Array(N * N * 4);
    if (!zeroed) for (let i = 0; i < N * N; i++) slopeValues.set(SLOPE, i * 4);
    const slope0 = texture(slopeValues, `ocean-oracle-slope0-${zeroed}`, 4);
    const eids = Compute.root.createBuffer(d.arrayOf(d.u32, 1)).$usage("storage");
    const transforms = Compute.root.createBuffer(d.arrayOf(Xform, 1)).$usage("storage");
    const vertices = Compute.root.createBuffer(d.arrayOf(d.vec4u, 1)).$usage("storage");
    const duskSky = Compute.root.createBuffer(DuskSkyGpu).$usage("uniform");
    const surfaceGroup = Compute.root.createBindGroup(oceanSurfaceLayout, {
        eids,
        transforms,
        displace0: displace0.createView(),
        displace1: displace1.createView(),
        slope0: slope0.createView(),
        slopeSampler: Compute.device.createSampler({ minFilter: "linear", mipmapFilter: "linear" }),
        duskSky,
        vertices,
    });
    const params = Compute.root.createBuffer(ProbeParams).$usage("uniform");
    params.write({ world: WORLD, slopeTexel: d.vec2i(4, 4) });
    const output = Compute.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const paramsGroup = Compute.root.createBindGroup(probeParamsLayout, { params });
    const outputGroup = Compute.root.createBindGroup(probeOutputLayout, {
        output: Compute.root.createBuffer(d.arrayOf(d.vec4f, 1), output).$usage("storage"),
    });
    const pipeline = Compute.root.createComputePipeline({ compute: kernel });
    const encoder = Compute.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pipeline
        .with(paramsGroup)
        .with(outputGroup)
        .with(surfaceGroup)
        .with(pass)
        .dispatchWorkgroups(1);
    pass.end();
    Compute.device.queue.submit([encoder.finish()]);
    const result = new Float32Array((await probeBuffer(Compute.device, output)).bytes);
    for (const resource of [displace0, displace1, slope0]) resource.destroy();
    for (const resource of [eids, transforms, vertices, duskSky, params]) resource.destroy();
    output.destroy();
    return result;
}

const F32_U = 2 ** -24;
const ESTIMATOR_OPS = 640;

export function oceanSurfaceCompiled(surfaces: Pick<typeof Surfaces, "has">): boolean {
    return surfaces.has("ocean");
}

export async function runDeviceClaim(state: State): Promise<Check[]> {
    state.pause();
    const reference = expected(false);
    const actual = await dispatch(false);
    const zero = await dispatch(true);
    const max = Math.max(...reference.map((value, i) => Math.abs(value - actual[i])));
    const gamma = (ESTIMATOR_OPS * F32_U) / (1 - ESTIMATOR_OPS * F32_U);
    const bound = gamma * Math.max(1, ...reference.map(Math.abs));
    const witness = Math.max(...reference.map((value, i) => Math.abs(value - zero[i])));
    const compiled = oceanSurfaceCompiled(Surfaces);
    const published = ["displace0", "displace1", "slope0"].every((name) =>
        Compute.textures.has(name),
    );
    return [
        {
            name: "registered ocean surface compiled and bound against published textures",
            pass: compiled && published,
            detail: `compiled=${compiled} published=${published}`,
        },
        {
            name: "device shipped estimator agrees with position-encoded closed form",
            pass: max <= bound,
            detail: `max=${max} bound=${bound} u=${F32_U} ops=${ESTIMATOR_OPS}`,
        },
        {
            name: "zeroed texture payload reds the same estimator comparison",
            pass: witness > bound,
            detail: `zeroedDeviation=${witness} sharedBound=${bound}`,
        },
    ];
}
