import {
    AmbientLight,
    Camera,
    Compute,
    font,
    GlazePlugin,
    InputPlugin,
    type Mirror,
    MirrorPlugin,
    mirror,
    Orbit,
    OrbitPlugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    SlabPlugin,
    type System,
    Text,
    TextPlugin,
    Transform,
    TransformsPlugin,
    text,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { Render, Views } from "@dylanebert/shallot/render/core";
import { ColorSystem } from "@dylanebert/shallot/sear/core";
import { type Check, frames, register, type Scenario, settle } from "../gym";

// The typed shallot text producer's real-GPU coverage: a label drawn through the `typedTextSurface`
// path (`packages/shallot/src/extras/text/index.ts`), gated the same way `accel`'s line draw is — a
// framebuffer probe that can only read positive if the surface actually put color on screen. `font()`
// runs before `run()` — `TextPlugin.initialize` only falls back to the remote `DEFAULT_FONT` when the
// registry is still empty at that point, so a font already registered wins.
const FONT_URL = "/font.ttf";

// out[0] = peak linear luminance over a 128×128 sample grid, out[1] = the minimum, out[2] = the count of
// sampled texels carrying real chroma — same shape as `accel.ts`'s line-draw probe. The camera's default
// clear is a near-neutral dark grey (luminance ≈ 0.023, channel spread ≈ 0.007), so a saturated label
// color lights up both peak and chroma only where glyphs actually rendered.
const CHROMA = 0.05;
const PROBE_WGSL = /* wgsl */ `
@group(0) @binding(0) var fb: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    let dim = vec2<f32>(textureDimensions(fb));
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let N = 128;
    var peak = 0.0;
    var lo = 1e9;
    var chroma = 0.0;
    for (var y = 0; y < N; y = y + 1) {
        for (var x = 0; x < N; x = x + 1) {
            let uv = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) / f32(N);
            let rgb = textureLoad(fb, vec2<i32>(uv * dim), 0).rgb;
            let lum = dot(rgb, luma);
            peak = max(peak, lum);
            lo = min(lo, lum);
            let spread = max(rgb.r, max(rgb.g, rgb.b)) - min(rgb.r, min(rgb.g, rgb.b));
            if (spread > ${CHROMA}) { chroma = chroma + 1.0; }
        }
    }
    out[0] = peak;
    out[1] = lo;
    out[2] = chroma;
}`;

let probePipeline: GPUComputePipeline | null = null;
let probeBuf: GPUBuffer | null = null;
let probeBg: GPUBindGroup | null = null;
let probeMirror: Mirror | null = null;

const ProbeSystem: System = {
    name: "text-probe",
    group: "draw",
    after: [ColorSystem],
    update(state) {
        if (!Render.encoder || !probePipeline || !probeBuf) return;
        for (const eid of state.query([Camera, Sear])) {
            const view = Views.get(eid);
            if (!view?.framebuffer) continue;
            if (!probeBg) {
                probeBg = Compute.device.createBindGroup({
                    label: "text-probe",
                    layout: probePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: view.framebuffer },
                        { binding: 1, resource: { buffer: probeBuf } },
                    ],
                });
            }
            const pass = Render.encoder.beginComputePass({ label: "text-probe" });
            pass.setPipeline(probePipeline);
            pass.setBindGroup(0, probeBg);
            pass.dispatchWorkgroups(1);
            pass.end();
            return;
        }
    },
};

// the typed text surface's positive framebuffer gate: a mesh/draw/pipeline resource count says a surface
// was registered, not that it put color on screen. Red-proven by mutating the surface's fs (forcing the
// discard, or zeroing alpha) — both drop the peak to the clear color and the chroma count to zero.
async function assertTextDraw(): Promise<Check> {
    if (!probeMirror) return { name: "text draw", pass: false, detail: "no probe mirror" };
    await settle(probeMirror);
    const snap = probeMirror.snapshot;
    if (!snap) return { name: "text draw", pass: false, detail: "no probe snapshot" };
    const out = new Float32Array(snap.bytes);
    const [peak, lo, chroma] = [out[0], out[1], out[2]];
    return {
        name: "text draw framebuffer",
        pass: peak > 0.2 && peak > lo * 4 && chroma > 0,
        detail: `peak ${peak.toFixed(4)}, min ${lo.toFixed(4)}, chroma texels ${chroma} / 16384`,
    };
}

const scenario: Scenario = {
    name: "text",
    async build(_canvas) {
        // register the local fixture font BEFORE run() — TextPlugin.initialize only falls back to the
        // remote DEFAULT_FONT when the registry is still empty at that point
        const fontId = font(FONT_URL, "gym-font");
        const contentId = text("Shallot");

        const { state, dispose } = await run({
            defaults: false,
            capacity: 16,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                SearPlugin,
                GlazePlugin,
                MirrorPlugin,
                TextPlugin,
            ],
        });

        const light = state.create();
        state.add(light, AmbientLight);

        const label = state.create();
        state.add(label, Transform);
        state.add(label, Text);
        Text.content.set(label, contentId);
        Text.font.set(label, fontId);
        Text.fontSize.set(label, 1);
        Text.anchor.set(label, 0.5, 0.5);
        Text.color.set(label, 0xff3366);

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.fov.set(cam, 55);
        Orbit.distance.set(cam, 3.2);
        Orbit.yaw.set(cam, 0);
        Orbit.pitch.set(cam, 0);

        await frames(1);

        probePipeline = await Compute.device.createComputePipelineAsync({
            label: "text-probe",
            layout: "auto",
            compute: {
                module: Compute.device.createShaderModule({
                    label: "text-probe",
                    code: PROBE_WGSL,
                }),
                entryPoint: "main",
            },
        });
        probeBuf = Compute.device.createBuffer({
            label: "text-probe",
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        probeBg = null;
        state.addSystem(ProbeSystem);
        await frames(4);
        probeMirror = mirror(probeBuf);
        await frames(3);

        return {
            state,
            dispose() {
                probeBuf?.destroy();
                probePipeline = null;
                probeBuf = null;
                probeBg = null;
                probeMirror = null;
                dispose();
            },
        };
    },
    async assert(): Promise<Check[]> {
        return [await assertTextDraw()];
    },
    live(): string {
        return "text label centered, straight-on camera";
    },
};

register(scenario);
