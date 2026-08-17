import {
    AmbientLight,
    Camera,
    CameraMode,
    Color,
    Compute,
    DirectionalLight,
    GlazePlugin,
    InputPlugin,
    type Mirror,
    MirrorPlugin,
    mirror,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    Shadow,
    SlabPlugin,
    type State,
    type System,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import {
    image,
    Profile,
    ProfilePlugin,
    Sprite,
    SpriteBillboard,
    SpriteBlend,
    SpritePlugin,
} from "@dylanebert/shallot/extras";
import { Draws, Render, Views } from "@dylanebert/shallot/render/core";
import { ColorSystem } from "@dylanebert/shallot/sear/core";
import { type Check, frames, type Params, register, type Scenario, settle } from "../gym";

// sprite — the SpritePlugin dogfood: procedurally-drawn icons (no asset files) amid cube Parts under a
// sun with shadows. Covers the three billboard modes (screen markers above the cubes, y-locked trees,
// a world-aligned ground decal), the opt-in alpha blend (a ghost marker), clip cutouts casting holed
// shadows, and the perspective↔ortho camera switch (a top-down consumer's shape — near
// straight-down, exercising the y-locked degeneracy guard's neighborhood). Gated on the per-bucket
// indirect instanceCount read back through a Mirror of the sprite arg buffer, a positive framebuffer
// chroma probe (the typed surface actually put color on screen — the text/accel precedent), and a
// snapshot-diff shadow probe that discriminates *this* caster's own cast from any other caster in the
// scene (billboard orientation and cutout shape are visual — read them in the screenshot, not here).

// the six (billboard, blend) buckets, billboard-major — mirrors extras/sprite's routing
const BUCKETS = 6;

// one canvas-drawn icon: a filled emblem on a transparent field, with a punched hole so the clip
// cutout (and its holed shadow) is visible. Returns a Blob image() accepts
async function icon(draw: (ctx: OffscreenCanvasRenderingContext2D) => void): Promise<Blob> {
    const canvas = new OffscreenCanvas(128, 128);
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, 128, 128);
    draw(ctx);
    return canvas.convertToBlob();
}

function house(ctx: OffscreenCanvasRenderingContext2D): void {
    ctx.fillStyle = "#e8e4da";
    ctx.fillRect(28, 56, 72, 56);
    ctx.beginPath();
    ctx.moveTo(16, 60);
    ctx.lineTo(64, 16);
    ctx.lineTo(112, 60);
    ctx.closePath();
    ctx.fill();
    ctx.clearRect(52, 76, 24, 36); // the door — a transparent hole the cutout (and shadow) keeps
}

function tree(ctx: OffscreenCanvasRenderingContext2D): void {
    ctx.fillStyle = "#5a8a4a";
    ctx.beginPath();
    ctx.moveTo(64, 8);
    ctx.lineTo(108, 88);
    ctx.lineTo(20, 88);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#7a5a3a";
    ctx.fillRect(56, 88, 16, 32);
}

function star(ctx: OffscreenCanvasRenderingContext2D): void {
    ctx.fillStyle = "#e8c84a";
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
        const r = i % 2 ? 24 : 56;
        const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
        ctx.lineTo(64 + Math.cos(a) * r, 64 + Math.sin(a) * r);
    }
    ctx.closePath();
    ctx.fill();
}

function box(
    state: State,
    pos: [number, number, number],
    scale: [number, number, number],
    color: [number, number, number],
): number {
    const eid = state.create();
    state.add(eid, Transform);
    Transform.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Transform.scale.set(eid, scale[0], scale[1], scale[2], 0);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

function sprite(
    state: State,
    img: number,
    pos: [number, number, number],
    fields: Partial<{
        billboard: number;
        blend: number;
        opacity: number;
        anchor: [number, number];
        size: [number, number];
    }> = {},
): number {
    const eid = state.create();
    state.add(eid, Transform);
    Transform.pos.set(eid, pos[0], pos[1], pos[2], 0);
    state.add(eid, Sprite);
    Sprite.image.set(eid, img);
    Sprite.size.set(eid, ...(fields.size ?? [1.4, 1.4]));
    if (fields.billboard !== undefined) Sprite.billboard.set(eid, fields.billboard);
    if (fields.blend !== undefined) Sprite.blend.set(eid, fields.blend);
    if (fields.opacity !== undefined) Sprite.opacity.set(eid, fields.opacity);
    if (fields.anchor) Sprite.anchor.set(eid, fields.anchor[0], fields.anchor[1]);
    return eid;
}

let cam = 0;
// a screen-clip marker the assert toggles: proves the live rebuild path, and doubles as
// `assertOwnShadow`'s caster (it already has a punched hole, so a working cast would be visibly holed)
let hideable = -1;
let spriteArgs: Mirror | null = null;
let params: Params | null = null;
let sceneState: State | null = null; // for `assertSpriteDraw`'s all-sprites visibility diff

// per-bucket instanceCount from the sprite indirect args (20-byte DrawIndexedIndirect records, count at word 1)
function bucketCounts(m: Mirror): number[] | null {
    if (!m.snapshot) return null;
    const args = new Uint32Array(m.snapshot.bytes);
    return Array.from({ length: BUCKETS }, (_, b) => args[b * 5 + 1]);
}

// the mode knob is projection only — it flips Camera.mode live without touching the orbit pose, so a
// live drag isn't fought. The ortho top-down pose is a build-time initial pose (see build)
function applyCamera(): void {
    const ortho = params?.mode === "ortho";
    Camera.mode.set(cam, ortho ? CameraMode.Orthographic : CameraMode.Perspective);
    Camera.size.set(cam, 7);
}

// the 128×128 sample grid's side, shared by both probes below
const GRID = 128;

// out[0] = peak linear luminance over the grid, out[1] = its minimum, out[2] = the count of sampled
// texels carrying real chroma, out[3] = the largest per-texel luminance *increase* against `snap`
// (a full-grid snapshot `mode` 0 writes and `mode` 1 diffs against) with its uv in out[4]/out[5] — the
// shadow-cast probe (`assertOwnShadow`): snapshotting while the caster casts, then diffing a later
// frame with it hidden, finds wherever hiding *that one sprite* brightened the frame, attributing the
// change to it specifically (an unrelated shadow is identical in both frames and diffs to ~0). out[6] =
// the largest per-texel luminance *magnitude* change either direction — the draw probe
// (`assertSpriteDraw`): a hidden sprite's own on-screen pixels change color (not necessarily brighter or
// darker — the ground/cube behind it might be either), so attributing the drop to sprite specifically
// needs the unsigned magnitude, not signed brightening.
const CHROMA = 0.05;
// the sample grid maps onto this uv sub-rect, not the whole frame — the object cluster (icons, trees,
// cubes) occupies roughly the center of the frame at this camera pose, and a small on-screen icon is
// easy for a sparse 128×128 sample of the *whole* frame to miss entirely; mapping the same sample count
// onto just the cluster's footprint raises the effective resolution there instead
const RECT_MIN = [0.2, 0.2];
const RECT_MAX = [0.85, 0.8];
const PROBE_WGSL = /* wgsl */ `
@group(0) @binding(0) var fb: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> snap: array<f32>;
@group(0) @binding(3) var<uniform> mode: u32;
@compute @workgroup_size(1)
fn main() {
    let dim = vec2<f32>(textureDimensions(fb));
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let N = ${GRID};
    let rectMin = vec2<f32>(${RECT_MIN[0]}, ${RECT_MIN[1]});
    let rectMax = vec2<f32>(${RECT_MAX[0]}, ${RECT_MAX[1]});
    var peak = 0.0;
    var lo = 1e9;
    var chroma = 0.0;
    var maxDiff = -1e9;
    var argU = 0.0;
    var argV = 0.0;
    var maxAbs = 0.0;
    for (var y = 0; y < N; y = y + 1) {
        for (var x = 0; x < N; x = x + 1) {
            let uv = mix(rectMin, rectMax, vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) / f32(N));
            let rgb = textureLoad(fb, vec2<i32>(uv * dim), 0).rgb;
            let lum = dot(rgb, luma);
            peak = max(peak, lum);
            lo = min(lo, lum);
            let spread = max(rgb.r, max(rgb.g, rgb.b)) - min(rgb.r, min(rgb.g, rgb.b));
            if (spread > ${CHROMA}) { chroma = chroma + 1.0; }
            let idx = y * N + x;
            if (mode == 0u) {
                snap[idx] = lum;
            } else {
                let diff = lum - snap[idx];
                if (diff > maxDiff) {
                    maxDiff = diff;
                    argU = uv.x;
                    argV = uv.y;
                }
                maxAbs = max(maxAbs, abs(diff));
            }
        }
    }
    out[0] = peak;
    out[1] = lo;
    out[2] = chroma;
    out[3] = maxDiff;
    out[4] = argU;
    out[5] = argV;
    out[6] = maxAbs;
}`;

let probePipeline: GPUComputePipeline | null = null;
let probeBuf: GPUBuffer | null = null;
let probeSnap: GPUBuffer | null = null;
let probeMode: GPUBuffer | null = null;
let probeBg: GPUBindGroup | null = null;
let probeMirror: Mirror | null = null;

const ProbeSystem: System = {
    name: "sprite-probe",
    group: "draw",
    after: [ColorSystem],
    update(state) {
        if (!Render.encoder || !probePipeline || !probeBuf || !probeSnap || !probeMode) return;
        for (const eid of state.query([Camera, Sear])) {
            const view = Views.get(eid);
            if (!view?.framebuffer) continue;
            if (!probeBg) {
                probeBg = Compute.device.createBindGroup({
                    label: "sprite-probe",
                    layout: probePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: view.framebuffer },
                        { binding: 1, resource: { buffer: probeBuf } },
                        { binding: 2, resource: { buffer: probeSnap } },
                        { binding: 3, resource: { buffer: probeMode } },
                    ],
                });
            }
            const pass = Render.encoder.beginComputePass({ label: "sprite-probe" });
            pass.setPipeline(probePipeline);
            pass.setBindGroup(0, probeBg);
            pass.dispatchWorkgroups(1);
            pass.end();
            return;
        }
    },
};

// the typed sprite surface's positive framebuffer gate: a mesh/draw/pipeline resource count says a
// surface was registered, not that it put color on screen. Snapshots the grid with every sprite
// visible, hides them all, and diffs — the largest *magnitude* change (not signed brightening: the
// ground/cube revealed behind a hidden icon can read either lighter or darker) is exactly the region
// hiding sprite geometry touched. The scene also carries Part cubes and a background, so a bare
// peak/chroma floor (the text/accel shape, whose scenes carry nothing else) would still pass with
// sprite's fs fully broken; this diff attributes the change to sprite specifically. Red-proven by
// mutating the fs (forcing every variant's cutoff to always discard): sprites already draw nothing, so
// hiding them changes nothing further and the magnitude collapses to noise.
async function assertSpriteDraw(): Promise<Check> {
    if (!probeMirror || !probeMode || !sceneState) {
        return { name: "sprite draw", pass: false, detail: "no probe mirror" };
    }
    const device = Compute.device;

    // snapshot the grid with every sprite visible (mode 0), then freeze it (mode 1) before hiding them
    device.queue.writeBuffer(probeMode, 0, new Uint32Array([0]));
    await frames(3);
    device.queue.writeBuffer(probeMode, 0, new Uint32Array([1]));
    await frames(2);

    const sprites = Array.from(sceneState.query([Sprite]));
    for (const eid of sprites) Sprite.visible.set(eid, 0);
    await settle(probeMirror);
    const snap = probeMirror.snapshot!;
    const out = new Float32Array(snap.bytes);
    const maxAbs = out[6];

    for (const eid of sprites) Sprite.visible.set(eid, 1);
    device.queue.writeBuffer(probeMode, 0, new Uint32Array([0]));
    await frames(2);

    return {
        name: "sprite draw framebuffer",
        pass: maxAbs > 0.03,
        detail: `max magnitude change ${maxAbs.toFixed(4)} (all sprites visible → hidden)`,
    };
}

// proves the dedicated caster's own shadow cast, attributable to that one sprite rather than any other
// shadow-casting geometry in the scene (the ground box included). Snapshots the sampled luminance grid
// while the caster casts, hides it, and diffs a later frame against that snapshot — the largest
// brightening is wherever hiding *this* sprite mattered, regardless of where on screen that is or what
// else in the scene also casts (an unrelated shadow is identical in both frames and diffs to ~0). A
// Part cube run through the same rig (scaled to zero) finds its shadow's exact tile with `maxDiff` ≈
// 0.33, the probe's own soundness proof.
async function assertOwnShadow(): Promise<Check> {
    if (!probeMirror || !probeMode || hideable < 0) {
        return { name: "sprite own shadow", pass: false, detail: "no probe mirror" };
    }
    const device = Compute.device;

    // snapshot the grid while the caster casts (mode 0), then freeze it (mode 1, diff-only — dispatch
    // keeps running every frame regardless, so the freeze must land BEFORE the caster hides or the
    // still-running mode-0 snapshot keeps overwriting `snap` with already-hidden frames)
    device.queue.writeBuffer(probeMode, 0, new Uint32Array([0]));
    await frames(3);
    device.queue.writeBuffer(probeMode, 0, new Uint32Array([1]));
    await frames(2);

    // hide the caster and diff the next frame against the frozen snapshot
    Sprite.visible.set(hideable, 0);
    await settle(probeMirror);
    const snap = probeMirror.snapshot!;
    const out = new Float32Array(snap.bytes);
    const [maxDiff, atU, atV] = [out[3], out[4], out[5]];

    Sprite.visible.set(hideable, 1);
    device.queue.writeBuffer(probeMode, 0, new Uint32Array([0]));
    await frames(2);

    return {
        name: "sprite own shadow cast (snapshot-diff)",
        pass: maxDiff > 0.03,
        detail: `max brightening ${maxDiff.toFixed(4)} at (${atU.toFixed(3)}, ${atV.toFixed(3)})`,
    };
}

const scenario: Scenario = {
    name: "sprite",
    params: [
        {
            key: "mode",
            type: "select",
            default: "perspective",
            options: ["perspective", "ortho"],
            label: "camera",
        },
    ],

    async build(_canvas: HTMLCanvasElement, p: Params) {
        params = p;
        sceneState = null;
        // register images before run() — SpritePlugin.initialize uploads the texture array
        const houseImg = image(await icon(house), "house");
        const treeImg = image(await icon(tree), "tree");
        const starImg = image(await icon(star), "star");

        const { state, dispose } = await run({
            defaults: false,
            capacity: 64,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
                MirrorPlugin,
                SpritePlugin,
            ],
        });
        sceneState = state;

        state.add(state.create(), AmbientLight);
        const sun = state.create();
        state.add(sun, DirectionalLight);
        DirectionalLight.direction.set(sun, -0.4, -0.8, -0.45, 0);
        state.add(sun, Shadow); // clip sprites cast holed shadows through sear's prepass pipeline

        // ground + a few cubes the icons sit amid (depth correctness reads against them)
        box(state, [0, -1.2, 0], [12, 0.4, 12], [0.26, 0.28, 0.3]);
        box(state, [-2.5, -0.5, -1], [1, 1, 1], [0.45, 0.4, 0.5]);
        box(state, [2.5, -0.5, 1], [1, 1, 1], [0.4, 0.5, 0.45]);

        // screen-aligned markers above the cubes (anchor 0.5 0 — the icon's base sits at the pos)
        hideable = sprite(state, houseImg, [-2.5, 0.2, -1], { anchor: [0.5, 0] });
        sprite(state, starImg, [2.5, 0.2, 1], { anchor: [0.5, 0] });
        sprite(state, starImg, [0, 0.4, -3], { anchor: [0.5, 0] });

        // y-locked trees (upright, yawing toward the camera)
        sprite(state, treeImg, [-1, -1, 2.5], {
            billboard: SpriteBillboard.YLocked,
            anchor: [0.5, 0],
        });
        sprite(state, treeImg, [1.5, -1, -2.8], {
            billboard: SpriteBillboard.YLocked,
            anchor: [0.5, 0],
        });

        // a world-aligned ground decal: rotated flat (-90° about X), riding the plain transform
        const decal = sprite(state, starImg, [0, -0.99, 1.8], { billboard: SpriteBillboard.World });
        Transform.rot.set(decal, -Math.SQRT1_2, 0, 0, Math.SQRT1_2);

        // the opt-in translucent mode: a half-faded ghost marker
        sprite(state, houseImg, [0.8, 0.2, 3], { blend: SpriteBlend.Alpha, opacity: 0.5 });

        cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.fov.set(cam, 55);
        Orbit.distance.set(cam, 11);
        Orbit.yaw.set(cam, 0.4);
        // ortho builds at the top-down framing — near straight-down (just under Orbit's
        // π/2 − 0.01 maxPitch clamp), the y-locked degeneracy guard's neighborhood. Initial pose only;
        // a live drag moves it. Perspective keeps the orbit default pitch
        if (p.mode === "ortho") Orbit.pitch.set(cam, Math.PI / 2 - 0.02);
        applyCamera();

        // the sprite draws register in SpriteSystem.setup (first frame); mirror the arg buffer after it
        await frames(1);
        spriteArgs = mirror(Draws.get("sprite-screen")!.args.indirect);

        probePipeline = await Compute.device.createComputePipelineAsync({
            label: "sprite-probe",
            layout: "auto",
            compute: {
                module: Compute.device.createShaderModule({
                    label: "sprite-probe",
                    code: PROBE_WGSL,
                }),
                entryPoint: "main",
            },
        });
        probeBuf = Compute.device.createBuffer({
            label: "sprite-probe",
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        probeSnap = Compute.device.createBuffer({
            label: "sprite-probe-snap",
            size: GRID * GRID * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        probeMode = Compute.device.createBuffer({
            label: "sprite-probe-mode",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        probeBg = null;
        state.addSystem(ProbeSystem);
        await frames(4);
        probeMirror = mirror(probeBuf);
        await frames(3);

        return {
            state,
            dispose() {
                cam = 0;
                hideable = -1;
                spriteArgs = null;
                sceneState = null;
                probeBuf?.destroy();
                probeSnap?.destroy();
                probeMode?.destroy();
                probePipeline = null;
                probeBuf = null;
                probeSnap = null;
                probeMode = null;
                probeBg = null;
                probeMirror = null;
                params = null;
                dispose();
            },
        };
    },

    // the gate: per-bucket indirect instanceCounts match the spawned set, the live rebuild path reacts
    // to a visibility edit, a positive framebuffer probe proves the typed surface put color on screen,
    // a snapshot-diff probe attributes a shadow specifically to one caster (see `assertOwnShadow`'s own
    // header for why it currently reads red for a sprite caster), and the counts hold under the ortho
    // top-down camera. Billboard orientation + cutout shape are visual (screenshot), not here
    async assert(): Promise<Check[]> {
        const checks: Check[] = [];
        if (!spriteArgs) return [{ name: "sprite: args mirror", pass: false, detail: "no mirror" }];

        const expected = [3, 1, 2, 0, 1, 0]; // screen, screen-alpha, y, y-alpha, world, world-alpha
        await settle(spriteArgs);
        let counts = bucketCounts(spriteArgs);
        checks.push({
            name: "sprite: per-bucket instance counts match the spawned set",
            pass: counts !== null && expected.every((n, b) => counts?.[b] === n),
            detail: `counts ${counts?.join(",") ?? "(none)"} (expected ${expected.join(",")})`,
        });

        Sprite.visible.set(hideable, 0);
        await settle(spriteArgs);
        counts = bucketCounts(spriteArgs);
        checks.push({
            name: "sprite: a visibility edit rebuilds the buckets live",
            pass: counts?.[0] === expected[0] - 1,
            detail: `screen bucket ${counts?.[0]} (expected ${expected[0] - 1})`,
        });
        Sprite.visible.set(hideable, 1);

        checks.push(await assertSpriteDraw());
        checks.push(await assertOwnShadow());

        checks.push({
            name: "sprite: the shadow map renders",
            pass: Profile.gpu.has("sear:cascadeshadow"),
            detail: `gpu passes: ${[...Profile.gpu.keys()].sort().join(", ")}`,
        });

        // the ortho top-down framing (the top-down consumer): the frame keeps rendering and the
        // bucket routing is camera-independent. Restore the URL-resolved mode after, so the
        // post-run screenshot shows the camera the run was asked for
        const prior = params!.mode;
        params!.mode = "ortho";
        applyCamera();
        await settle(spriteArgs);
        counts = bucketCounts(spriteArgs);
        checks.push({
            name: "sprite: ortho top-down camera holds the same buckets",
            pass: counts !== null && expected.every((n, b) => counts?.[b] === n),
            detail: `counts ${counts?.join(",") ?? "(none)"} (expected ${expected.join(",")})`,
        });
        params!.mode = prior;
        applyCamera();
        await frames(2);
        return checks;
    },

    live(): string {
        applyCamera(); // the camera select is a live knob — re-applied so the dropdown takes effect
        return "sprite — drag to orbit; screen markers, y-locked trees, a ground decal, a ghost (alpha)";
    },
};

register(scenario);
