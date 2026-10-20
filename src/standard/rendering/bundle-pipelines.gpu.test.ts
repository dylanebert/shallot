import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import {
    compileGpuFile,
    createTestGpuApp,
    disposeGpuApps,
    gpuRequirements,
} from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    DepthPrepass,
    DirectionalLight,
    PointLight,
    SpotLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation, type World } from "../../engine";
import { rawDevice } from "../../engine/runtime";
import { StandardRenderer } from "./forward";
import { MaterialPlugin, materialType, StandardRenderingPlugin } from "./index";
import { AlphaMode, Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";

setDefaultTimeout(TIMEOUT.gpu);

const VaryingParameters = d.struct({ color: d.vec4f });
const VaryingMaterial = materialType({
    name: "BundleVaryingMaterial",
    parameters: VaryingParameters,
    varyings: { tint: d.vec4f },
    vertex: (input) => {
        "use gpu";
        return { world: input.world, worldNormal: input.worldNormal, tint: input.parameters.color };
    },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.tint);
    },
    defaults: { color: d.vec4f(1) },
});

const plugins = [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(VaryingMaterial)];

// The alpha keys whose pipelines draw opaque, and so are recorded into bundles
const OPAQUE = 0;
const MASK = 1;
const ALPHA_TO_COVERAGE = 5;
const KEYS = [OPAQUE, MASK, ALPHA_TO_COVERAGE];
const MODES = [AlphaMode.Opaque, AlphaMode.Mask(0.5), AlphaMode.AlphaToCoverage];
const SURFACES = ["StandardMaterial", "BundleVaryingMaterial"];

type Bundled = { label: string; fragment: boolean };

// WebKit fails `setPipeline` on a bundle encoder given a pipeline with no fragment stage, though Dawn
// and the spec accept it; this device records each bundled pipeline so the test holds WebKit's rule.
function bundleRecorder(device: GPUDevice): { into: Bundled[] } {
    const fragmentless = new WeakSet<GPURenderPipeline>();
    const sink = { into: [] as Bundled[] };
    const create = device.createRenderPipeline.bind(device);
    device.createRenderPipeline = (descriptor) => {
        const pipeline = create(descriptor);
        if (!descriptor.fragment) fragmentless.add(pipeline);
        return pipeline;
    };
    const createAsync = device.createRenderPipelineAsync.bind(device);
    device.createRenderPipelineAsync = async (descriptor) => {
        const pipeline = await createAsync(descriptor);
        if (!descriptor.fragment) fragmentless.add(pipeline);
        return pipeline;
    };
    const createBundle = device.createRenderBundleEncoder.bind(device);
    device.createRenderBundleEncoder = (descriptor) => {
        const encoder = createBundle(descriptor);
        const setPipeline = encoder.setPipeline.bind(encoder);
        encoder.setPipeline = (pipeline) => {
            sink.into.push({ label: pipeline.label, fragment: !fragmentless.has(pipeline) });
            setPipeline(pipeline);
        };
        return encoder;
    };
    return sink;
}

function camera(
    world: World,
    config: { antialias: number; order?: number },
    target?: ReturnType<typeof attachTexture>,
) {
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, 0, 8, 0] });
    world.add(eid, Camera, config);
    world.add(eid, AmbientLight, { brightness: 0.2 });
    world.add(eid, StandardRenderer);
    world.add(eid, DepthPrepass);
    return attachTexture(world, eid, target ?? { width: 32, height: 32 });
}

function surfaces(world: World): void {
    let x = -2.5;
    for (const mode of MODES) {
        for (const handle of [
            world.resource(Materials).add(StandardMaterial(), { alphaMode: mode }),
            world.resource(VaryingMaterial).add({ color: d.vec4f(1) }, { alphaMode: mode }),
        ]) {
            const eid = world.create();
            world.add(eid, Transform, { translation: [x, 0, 0, 0], scale: [0.4, 0.4, 0.4, 0] });
            world.add(eid, MeshInstance);
            world.add(eid, MeshMaterial, handle);
            x += 1;
        }
    }
}

async function frames(world: World, device: GPUDevice): Promise<string | null> {
    device.pushErrorScope("validation");
    world.step(0);
    world.step(0);
    return (await device.popErrorScope())?.message ?? null;
}

const subjects = compileGpuFile(import.meta.path, async () => {
    const owner = await createTestGpuApp({
        defaults: false,
        plugins: [{ name: "BundlePipelinesTestDevice", gpu: gpuRequirements(plugins) }],
    });
    const device = rawDevice(owner.world.gpu.device);
    const sink = bundleRecorder(device);

    // AA on and off with a prepass each, and an AA camera stacked over another on one target;
    // a shadowed sun and point light
    const scene: Bundled[] = (sink.into = []);
    const lit = await createTestGpuApp({ defaults: false, plugins, device });
    camera(lit.world, { antialias: 1 });
    camera(lit.world, { antialias: 0 });
    const lower = camera(lit.world, { antialias: 1 });
    camera(lit.world, { antialias: 1, order: 1 }, lower);
    surfaces(lit.world);
    const sun = lit.world.create();
    lit.world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.3, -0.3, -1) });
    lit.world.add(sun, DirectionalLight, { illuminance: 9406.831, shadowMapsEnabled: 1 });
    const point = lit.world.create();
    lit.world.add(point, Transform, { translation: [0, 1, 3, 0] });
    lit.world.add(point, PointLight, { intensity: 788064.8, range: 20, shadowMapsEnabled: 1 });
    const sceneError = await frames(lit.world, device);

    // a spot light is the only shadow caster
    const spotted: Bundled[] = (sink.into = []);
    const spot = await createTestGpuApp({ defaults: false, plugins, device });
    camera(spot.world, { antialias: 1 });
    surfaces(spot.world);
    const light = spot.world.create();
    spot.world.add(light, Transform, { translation: [0, 0, 4, 0] });
    spot.world.add(light, SpotLight, { intensity: 788064.8, range: 20, shadowMapsEnabled: 1 });
    const spotError = await frames(spot.world, device);

    return { owner, lit, spot, scene, sceneError, spotted, spotError };
});

function families(): string[] {
    const out: string[] = [];
    for (const surface of SURFACES) {
        for (const key of KEYS) {
            out.push(
                `standard-${surface}-${key}`,
                `standard-${surface}-${key}-1x`,
                `standard-prepass-${surface}-${key}-4x`,
                `standard-prepass-${surface}-${key}-1x`,
                `standard-cascade-${surface}-${key}`,
                `standard-point-${surface}-${key}`,
            );
        }
        out.push(`standard-${surface}-${ALPHA_TO_COVERAGE}-stacked`);
    }
    return out;
}

test("every pipeline family standard records into a render bundle has a fragment stage", () => {
    const { scene, sceneError, spotted, spotError } = subjects();
    expect(sceneError).toBeNull();
    expect(spotError).toBeNull();
    const reached = new Set(scene.map((step) => step.label));
    expect(families().filter((label) => !reached.has(label))).toEqual([]);
    const spotReached = new Set(spotted.map((step) => step.label));
    const spotFamilies = SURFACES.flatMap((s) => KEYS.map((k) => `standard-point-${s}-${k}`));
    expect(spotFamilies.filter((label) => !spotReached.has(label))).toEqual([]);
    expect([...scene, ...spotted].filter((step) => !step.fragment)).toEqual([]);
});

afterAll(() => {
    const { owner, lit, spot } = subjects();
    return disposeGpuApps([owner, lit, spot]);
});
