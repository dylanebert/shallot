// Editor-local viewport chrome — the grid + selection outline. Both are editor concerns, not engine
// API, so they live in the editor app rather than `@dylanebert/shallot`. Each is a render pass into the
// camera's offscreen scene-color (`view.framebuffer`) in the `after: [ColorSystem], before: [GlazeSystem]`
// seam, so glaze still presents the framebuffer with the chrome composited in. The grid depth-tests
// against the prepass depth (`view.depth`, camera carries `Depth`); the outline samples the id lane
// (`view.tag`, camera carries `Tag`). Ported from the legacy `extras/gizmos` + `extras/outline` shaders,
// adapted from the old deferred-renderer overlay/compute-graph to the forward renderer's per-view targets.
import {
    Camera,
    Compute,
    Depth,
    invert,
    type Plugin,
    RenderPlugin,
    Sear,
    type State,
    type System,
    sparse,
    Tag,
    Transform,
    u32,
    unpackColor,
} from "@dylanebert/shallot";
import { GlazeSystem } from "@dylanebert/shallot/glaze";
import { computeViewProj, Render, type View, Views } from "@dylanebert/shallot/render/core";
import { ColorSystem, DEPTH_FORMAT } from "@dylanebert/shallot/sear/core";
import {
    cursorRay,
    decodeHandle,
    glyphs,
    handleSegments,
    type Manipulator,
    PLANE_EDGE_FADE,
    PLANE_EDGE_MIN,
    Scale,
    type Vec3,
    WORLD_AXES,
} from "./gizmo";
import { current, type Palette, packed, rgb } from "./theme";

// the editor's viewport overlays decompose into three orthogonal axes (roadmap "decompose the mode
// annotation"): lifecycle (`annotations.mode` — when a system runs, read by the engine scheduler),
// layer (`annotations.layer: "tooling"` — editor-only, never shipped; the host composes it in, see
// plugins.ts `isTooling`), and category (`annotations.category` + the per-camera set below — what
// shows where). lifecycle stays the engine's; layer + category live in the open `annotations` bag the
// editor reads, so the engine core never learns an editor concept.

/** editor overlay categories — bit flags. each overlay system declares one via `annotations.category`;
 * a camera's {@link Overlays} set enables a subset, and an overlay draws into a camera iff its bit is set */
export const Overlay = { Grid: 1, Outline: 2 } as const;

/** per-camera enabled-overlay set — a bitmask of {@link Overlay} flags */
export const Overlays = {
    enabled: sparse(u32),
};

/** the cameras whose overlay set enables `category` — the per-view gate every overlay system shares */
export function overlayCameras(state: State, category: number): number[] {
    const out: number[] = [];
    for (const eid of state.query([Camera, Overlays])) {
        if ((Overlays.enabled.get(eid) & category) !== 0) out.push(eid);
    }
    return out;
}

// the prepass lane each overlay category draws against: grid depth-tests for occlusion (`view.depth`,
// marker Depth), outline samples the id lane (`view.tag`, marker Tag). A camera enabling a category but
// missing the marker runs no prepass for that lane, so `view.*` stays null and the overlay silently bails
// at its draw guard — the bug per-view composition has to close. The lane is a *declared dependency* of
// the category, not an assumption: setOverlays pulls the marker in when the category turns on.
const OVERLAY_LANES: { category: number; lane: object }[] = [
    { category: Overlay.Grid, lane: Depth },
    { category: Overlay.Outline, lane: Tag },
];

/** enable an overlay set on a camera, pulling in each enabled category's prepass-lane marker (Depth for
 * grid, Tag for outline) so the prepass fills the `view.*` the overlay reads. The lane is a declared
 * dependency of the category — added here, never assumed present — so a freshly-bound game camera that
 * enables an overlay draws it, the `!view.depth` / `!view.tag` guard satisfied. Additive: a category
 * toggled off keeps its lane (an idle prepass store, cheaper than churning the camera's targets). The
 * single seam every host call routes through — view setup, play-mode bind, and the toggle. */
export function setOverlays(state: State, eid: number, mask: number): void {
    if (!state.has(eid, Overlays)) state.add(eid, Overlays);
    Overlays.enabled.set(eid, mask);
    for (const { category, lane } of OVERLAY_LANES) {
        if ((mask & category) !== 0 && !state.has(eid, lane)) state.add(eid, lane);
    }
}

// grid uniform: viewProj (unproject for the ray) + invViewProj (screen corners → world) + camera world
// position (the distance-fade center). camPos comes straight off Transform.pos — scenes are flat, so the
// camera's world translation is its local one.
// mat4 viewProj (16) + mat4 invViewProj (16) + vec4 camPos (4) + vec4 grid/axisX/axisZ colors (12)
const GRID_UNIFORM_FLOATS = 48;
const GRID_UNIFORM_BYTES = GRID_UNIFORM_FLOATS * 4;

const GRID_SHADER = /* wgsl */ `
struct Grid {
    viewProj: mat4x4<f32>,
    invViewProj: mat4x4<f32>,
    camPos: vec4<f32>,
    gridColor: vec4<f32>,
    axisXColor: vec4<f32>,
    axisZColor: vec4<f32>,
}

@group(0) @binding(0) var<uniform> grid: Grid;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) nearPoint: vec3<f32>,
    @location(1) farPoint: vec3<f32>,
}

fn unproject(p: vec3<f32>) -> vec3<f32> {
    let u = grid.invViewProj * vec4(p, 1.0);
    return u.xyz / u.w;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
    let pos = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    );
    let p = pos[vi];
    var out: VSOut;
    out.position = vec4(p, 0.0, 1.0);
    out.nearPoint = unproject(vec3(p, 0.0));
    out.farPoint = unproject(vec3(p, 1.0));
    return out;
}

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

fn line(worldPos: vec3<f32>, scale: f32) -> f32 {
    let coord = worldPos.xz / scale;
    let d = fwidth(coord);
    let g = abs(fract(coord - 0.5) - 0.5) / d;
    return 1.0 - min(min(g.x, g.y), 1.0);
}

@fragment
fn fs(input: VSOut) -> FragOut {
    let t = -input.nearPoint.y / (input.farPoint.y - input.nearPoint.y);
    if (t < 0.0) { discard; }

    let worldPos = input.nearPoint + t * (input.farPoint - input.nearPoint);

    let clip = grid.viewProj * vec4(worldPos, 1.0);
    let depth = clip.z / clip.w;
    if (depth < 0.0 || depth > 1.0) { discard; }

    let dist = length(worldPos.xz - grid.camPos.xz);
    let fade = 1.0 - smoothstep(20.0, 80.0, dist);
    if (fade <= 0.0) { discard; }

    let minor = line(worldPos, 1.0);
    let major = line(worldPos, 10.0);
    let l = max(minor * 0.08, major * 0.13);
    if (l < 0.01) { discard; }

    var color = grid.gridColor.rgb;
    var alpha = l * fade;

    let aw = fwidth(worldPos.xz);
    let xAxis = 1.0 - min(abs(worldPos.z) / aw.y, 1.0);
    let zAxis = 1.0 - min(abs(worldPos.x) / aw.x, 1.0);
    if (xAxis > 0.01) {
        color = mix(color, grid.axisXColor.rgb, xAxis);
        alpha = max(alpha, xAxis * 0.8 * fade);
    }
    if (zAxis > 0.01) {
        color = mix(color, grid.axisZColor.rgb, zAxis);
        alpha = max(alpha, zAxis * 0.8 * fade);
    }

    var out: FragOut;
    out.depth = depth;
    out.color = vec4(color, alpha);
    return out;
}
`;

const ALPHA_BLEND: GPUBlendState = {
    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
};

const _grid: {
    pipeline: GPURenderPipeline | null;
    layout: GPUBindGroupLayout | null;
    buffer: GPUBuffer | null;
    bindGroup: GPUBindGroup | null;
} = { pipeline: null, layout: null, buffer: null, bindGroup: null };

const _gridData = new Float32Array(GRID_UNIFORM_FLOATS);
const _viewProj = new Float32Array(16);
const _invViewProj = new Float32Array(16);
// the grid/axis colors sit in the same uniform but change only on a theme switch — written when the
// palette identity changes, not every frame like the camera matrices
let _gridPalette: Palette | null = null;

function drawGrid(eid: number, view: View): void {
    const { device } = Compute;
    const encoder = Render.encoder;
    if (!device || !encoder || !_grid.pipeline || !_grid.layout || !_grid.buffer) return;
    // the pipeline has a depthStencil state, so the pass MUST carry a depth attachment — and the grid
    // depth-tests for occlusion regardless. Skip when the camera has no Depth lane rather than encode a
    // pass that mismatches the pipeline (a per-frame validation error otherwise)
    if (!view.framebuffer || !view.depth || view.width === 0 || view.height === 0) return;

    computeViewProj(eid, view.width / view.height, _viewProj);
    invert(_viewProj, _invViewProj);
    _gridData.set(_viewProj, 0);
    _gridData.set(_invViewProj, 16);
    _gridData[32] = Transform.pos.x.get(eid);
    _gridData[33] = Transform.pos.y.get(eid);
    _gridData[34] = Transform.pos.z.get(eid);
    if (_gridPalette !== current.palette) {
        _gridPalette = current.palette;
        _gridData.set(rgb(current.palette.grid), 36);
        _gridData.set(rgb(current.palette.axisX), 40);
        _gridData.set(rgb(current.palette.axisZ), 44);
    }
    device.queue.writeBuffer(_grid.buffer, 0, _gridData);

    if (!_grid.bindGroup) {
        _grid.bindGroup = device.createBindGroup({
            label: "editor-grid",
            layout: _grid.layout,
            entries: [{ binding: 0, resource: { buffer: _grid.buffer } }],
        });
    }

    const pass = encoder.beginRenderPass({
        label: "editor-grid",
        colorAttachments: [{ view: view.framebuffer, loadOp: "load", storeOp: "store" }],
        // tests against the prepass depth so geometry occludes the grid; depthWriteEnabled is off in the
        // pipeline, so the stored depth is read-only in effect
        depthStencilAttachment: { view: view.depth, depthLoadOp: "load", depthStoreOp: "store" },
        timestampWrites: Compute.span?.("editor:grid"),
    });
    pass.setPipeline(_grid.pipeline);
    pass.setBindGroup(0, _grid.bindGroup);
    pass.draw(6);
    pass.end();
}

const GridSystem: System = {
    name: "editor-grid",
    group: "draw",
    annotations: { mode: "always", layer: "tooling", category: Overlay.Grid },
    after: [ColorSystem],
    before: [GlazeSystem],
    update(state: State) {
        for (const eid of overlayCameras(state, Overlay.Grid)) {
            const view = Views.get(eid);
            if (view) drawGrid(eid, view);
        }
    },
};

async function warmGrid(device: GPUDevice): Promise<void> {
    const module = device.createShaderModule({ label: "editor-grid", code: GRID_SHADER });
    _grid.layout = device.createBindGroupLayout({
        label: "editor-grid",
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            },
        ],
    });
    _grid.buffer?.destroy();
    _grid.buffer = device.createBuffer({
        label: "editor-grid",
        size: GRID_UNIFORM_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _grid.bindGroup = null;
    _grid.pipeline = await device.createRenderPipelineAsync({
        label: "editor-grid",
        layout: device.createPipelineLayout({ bindGroupLayouts: [_grid.layout] }),
        vertex: { module, entryPoint: "vs" },
        fragment: {
            module,
            entryPoint: "fs",
            targets: [{ format: Render.format, blend: ALPHA_BLEND }],
        },
        depthStencil: {
            format: DEPTH_FORMAT,
            depthCompare: "greater-equal",
            depthWriteEnabled: false,
        },
        primitive: { topology: "triangle-list" },
    });
}

function disposeGrid(): void {
    _grid.buffer?.destroy();
    _grid.buffer = null;
    _grid.pipeline = null;
    _grid.layout = null;
    _grid.bindGroup = null;
}

/** selection outline — the editor sets `getEntities` to the selected eids, the color, and the px width */
export interface Outline {
    getEntities: () => number[];
    color: number;
    thickness: number;
}

export const Outline: Outline = {
    getEntities: () => [],
    color: packed(current.palette.outline),
    thickness: 1,
};

const MAX_SELECTED = 256;

// outline = orrstead's coverage outline (orrstead/outline), reformulated as coverage-as-signed-distance
// analytic antialiasing so its one constant is derived, not tuned. The tag lane is a single-sample integer
// id — it can't MSAA-resolve, so its silhouette is a hard 1-bit staircase no MSAA touches. Averaging
// selection membership over a (2r+1)² box gives a coverage m that crosses 0.5 at the silhouette; the many
// samples make it finely fractional, which is the smoothing. A box filter's coverage changes by 1/N per
// pixel (N = 2r+1, the kernel width), so d = N·(m − 0.5) is the signed distance to the silhouette in pixels
// — N is the inverse coverage gradient, derived. The band saturate(N/2 − |d|) is then an analytic-AA
// threshold in Ben Golus's form (saturate(width − dist + bias)); N/2 is half the kernel support, the widest
// distance the box can resolve. This is algebraically identical to orrstead's saturate((1 − |2m−1|)·gain):
// substituting d gives saturate((1 − |2m−1|)·N/2), so the gain *is* N/2 — orrstead's 1.5 is exactly 3/2 at
// its 3×3, a kernel constant, not a magic number, and the validated look is unchanged. The whole selection
// reads as one silhouette (every selected eid counts toward m, so adjacent selections merge with no seam).
// A fullscreen pass alpha-blends the band into the framebuffer, so it never reads the color target it writes
// (the blend unit composites it).
const OUTLINE_SHADER = /* wgsl */ `
struct Uniforms {
    color: vec3<f32>,
    radius: i32,
    selectedCount: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var tag: texture_2d<u32>;
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var<storage, read> selected: array<u32>;

fn isSelected(eid: u32) -> bool {
    for (var i = 0u; i < u.selectedCount; i++) {
        if (selected[i] == eid) { return true; }
    }
    return false;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(tag));
    let coord = vec2<i32>(fragPos.xy);
    let n = f32(2 * u.radius + 1);

    // box coverage of the selection: fraction of the (2r+1)² neighbourhood whose tag is selected
    var hits = 0.0;
    for (var dy = -u.radius; dy <= u.radius; dy++) {
        for (var dx = -u.radius; dx <= u.radius; dx++) {
            let nc = clamp(coord + vec2<i32>(dx, dy), vec2<i32>(0), dims - 1);
            if (isSelected(textureLoad(tag, nc, 0).r)) { hits += 1.0; }
        }
    }
    let m = hits / (n * n);

    // m → signed distance to the silhouette (px), then the analytic-AA band. N (= 1/coverage-gradient) and
    // N/2 (the band half-width, orrstead's "EDGE_GAIN") are both derived from the kernel size — no tuning.
    let d = n * (m - 0.5);
    let alpha = saturate(n * 0.5 - abs(d));
    if (alpha <= 0.001) { discard; }
    return vec4(u.color, alpha);
}
`;

const _outline: {
    pipeline: GPURenderPipeline | null;
    layout: GPUBindGroupLayout | null;
    uniform: GPUBuffer | null;
    selectedBuf: GPUBuffer | null;
    bindGroup: GPUBindGroup | null;
    cachedTag: GPUTexture | null;
} = {
    pipeline: null,
    layout: null,
    uniform: null,
    selectedBuf: null,
    bindGroup: null,
    cachedTag: null,
};

const _outlineData = new ArrayBuffer(32);
const _outlineF32 = new Float32Array(_outlineData);
const _outlineI32 = new Int32Array(_outlineData);
const _outlineU32 = new Uint32Array(_outlineData);
const _selectedScratch = new Uint32Array(MAX_SELECTED);

function drawOutline(view: View): void {
    const { device } = Compute;
    const encoder = Render.encoder;
    if (
        !device ||
        !encoder ||
        !_outline.pipeline ||
        !_outline.layout ||
        !_outline.uniform ||
        !_outline.selectedBuf
    )
        return;
    if (!view.framebuffer || !view.tag) return;

    const entities = Outline.getEntities();
    if (entities.length === 0) return;

    const { r, g, b } = unpackColor(Outline.color);
    _outlineF32[0] = r;
    _outlineF32[1] = g;
    _outlineF32[2] = b;
    // box radius of the coverage kernel — band half-width is N/2 = radius + 0.5 px (orrstead's is radius 1)
    _outlineI32[3] = Math.max(1, Math.round(Outline.thickness));
    const count = Math.min(entities.length, MAX_SELECTED);
    _outlineU32[4] = count;
    device.queue.writeBuffer(_outline.uniform, 0, _outlineData);
    for (let i = 0; i < count; i++) _selectedScratch[i] = entities[i];
    device.queue.writeBuffer(_outline.selectedBuf, 0, _selectedScratch, 0, count);

    // the prepass recreates view.tag on resize; rebuild the bind group only when that texture changes
    if (!_outline.bindGroup || view.tag !== _outline.cachedTag) {
        _outline.bindGroup = device.createBindGroup({
            label: "editor-outline",
            layout: _outline.layout,
            entries: [
                { binding: 0, resource: view.tag.createView() },
                { binding: 1, resource: { buffer: _outline.uniform } },
                { binding: 2, resource: { buffer: _outline.selectedBuf } },
            ],
        });
        _outline.cachedTag = view.tag;
    }

    const pass = encoder.beginRenderPass({
        label: "editor-outline",
        colorAttachments: [{ view: view.framebuffer, loadOp: "load", storeOp: "store" }],
        timestampWrites: Compute.span?.("editor:outline"),
    });
    pass.setPipeline(_outline.pipeline);
    pass.setBindGroup(0, _outline.bindGroup);
    pass.draw(3);
    pass.end();
}

const OutlineSystem: System = {
    name: "editor-outline",
    group: "draw",
    annotations: { mode: "always", layer: "tooling", category: Overlay.Outline },
    after: [ColorSystem],
    before: [GlazeSystem],
    update(state: State) {
        for (const eid of overlayCameras(state, Overlay.Outline)) {
            const view = Views.get(eid);
            if (view) drawOutline(view);
        }
    },
};

async function warmOutline(device: GPUDevice): Promise<void> {
    const module = device.createShaderModule({ label: "editor-outline", code: OUTLINE_SHADER });
    _outline.layout = device.createBindGroupLayout({
        label: "editor-outline",
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "uint" },
            },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            {
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: "read-only-storage" },
            },
        ],
    });
    _outline.uniform?.destroy();
    _outline.uniform = device.createBuffer({
        label: "editor-outline-uniform",
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _outline.selectedBuf?.destroy();
    _outline.selectedBuf = device.createBuffer({
        label: "editor-outline-selected",
        size: MAX_SELECTED * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _outline.bindGroup = null;
    _outline.cachedTag = null;
    _outline.pipeline = await device.createRenderPipelineAsync({
        label: "editor-outline",
        layout: device.createPipelineLayout({ bindGroupLayouts: [_outline.layout] }),
        vertex: { module, entryPoint: "vs" },
        fragment: {
            module,
            entryPoint: "fs",
            targets: [{ format: Render.format, blend: ALPHA_BLEND }],
        },
        primitive: { topology: "triangle-list" },
    });
}

function disposeOutline(): void {
    _outline.uniform?.destroy();
    _outline.selectedBuf?.destroy();
    _outline.uniform = null;
    _outline.selectedBuf = null;
    _outline.pipeline = null;
    _outline.layout = null;
    _outline.bindGroup = null;
    _outline.cachedTag = null;
    Outline.getEntities = () => [];
    Outline.color = packed(current.palette.outline);
    Outline.thickness = 1;
}

/** the transform gizmo's host seam: the editor sets `getOrigin` (the selection's gizmo anchor in world,
 * or null to hide it), `getAxes` (the active frame — world, or the selection's local frame), `getManip`
 * (the active tool's {@link Manipulator}, whose handle set drives both draw and pick, or null to hide it —
 * read live like `getOrigin` so a tool switch retints without a rebuild), `active`/`hover` (the grabbed /
 * hovered handle id, -1 for none). {@link HandleSystem} draws it in edit mode only — the same world-space
 * geometry the hit-test grabs, sized for a constant on-screen extent from the camera distance. */
export interface Handles {
    getOrigin: () => Vec3 | null;
    getAxes: () => readonly [Vec3, Vec3, Vec3];
    getManip: () => Manipulator | null;
    /** the gizmo's world-space size for a constant on-screen extent — App computes it from the canvas's CSS
     * height so render and the CSS-px hit-test agree on every DPR (0 to hide). */
    getScale: () => number;
    active: number;
    hover: number;
}

export const Handles: Handles = {
    getOrigin: () => null,
    getAxes: () => WORLD_AXES,
    getManip: () => null,
    getScale: () => 0,
    active: -1,
    hover: -1,
};

// the solid glyphs' sizes (device px) — caps, arrowheads, plane fill. Linear handles are world-space lines
// (see HANDLE_LINE_SHADER), so the triangle scratch only holds the solids now.
const HANDLE_CAP = 5; // scale tip box / center point
const CUBE_PX = 8; // the scale centre handle half-extent — bigger than a tip cap so it reads as the primary grab
const ARROW_LEN = 11; // arrowhead length past the bar tip
const ARROW_HALF = 5; // arrowhead base half-width
const PLANE_ALPHA = 0.22; // plane quad fill, idle
const DISC_ALPHA = 0.07; // the free-rotate interior fill — shown only under the cursor, none at rest
const DISC_SEGS = 40; // triangle-fan segments for the free-rotate disc
const IDLE_ALPHA = 1.0;
const HANDLE_FLOATS = 4096; // solids only — arrowheads + caps + plane fills + centre dot + free-rotate disc

const HANDLE_SHADER = /* wgsl */ `
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VSOut {
    var out: VSOut;
    out.position = vec4(pos, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(input: VSOut) -> @location(0) vec4<f32> {
    return input.color;
}
`;

const _handles: { pipeline: GPURenderPipeline | null; buffer: GPUBuffer | null } = {
    pipeline: null,
    buffer: null,
};

const _handleData = new Float32Array(HANDLE_FLOATS);
const _handleVP = new Float32Array(16);
const _handleInv = new Float32Array(16);
// axis + neutral colors, reparsed only on a theme switch (palette identity), the way the grid caches its own
const _handleRGB = new Float32Array(9);
const _handleNeutral = new Float32Array(3);
const _handleGrid = new Float32Array(3); // grid-line color — the secondary handles (roll ring, free disc)
let _handlePalette: Palette | null = null;

// The linear handles (axis shafts, rings, plane-square edges) render as world-space lines through the
// proven `extras/lines` technique: the VS projects both endpoints, near-plane-clips, and expands a quad by
// a constant pixel half-width from `view.resolution` (DPR-correct), AA'd in the FS. This is what keeps them
// stable + correctly foreshortened at every angle — the hand-rolled screen projection couldn't. The solid
// glyphs (arrowheads, scale caps, centre, plane fill) stay on the triangle pipeline above, drawn at their
// projected points.
const LINE_W = 3.2; // line width, device px
const LINE_W_ON = 4.6; // hovered / active
// the rotate trackball (screen-facing ring) is secondary to the axis rings — drawn thin so it never
// overshadows them, rising to the normal width only on hover / active
const TRACK_W = 1.8;
const TRACK_W_ON = 3.2;
const TRACK_ALPHA = 0.4; // the roll ring is faint/secondary (grid-like); +state factor firms it under the cursor
const MAX_HANDLE_SEGS = 256; // 4 rings × 48 (Rotate: 3 axis + 1 trackball) is the busiest tool, with margin
const _segData = new Float32Array(MAX_HANDLE_SEGS * 8); // per seg: a.xyz, width, b.xyz, color(u32 bits)
const _segU32 = new Uint32Array(_segData.buffer);
const _lineUni = new Float32Array(20); // mat4 viewProj (16) + vec4 resolution (xy used)

const HANDLE_LINE_SHADER = /* wgsl */ `
struct Seg { a: vec3<f32>, w: f32, b: vec3<f32>, color: u32 }
struct Uni { viewProj: mat4x4<f32>, res: vec4<f32> }
@group(0) @binding(0) var<uniform> uni: Uni;
@group(0) @binding(1) var<storage, read> segs: array<Seg>;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) rgba: vec4<f32>,
    @location(1) edgeDist: f32,
    @location(2) halfPx: f32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) iid: u32) -> VSOut {
    // canonical quad as (t, edge) per vertex — two triangles
    var corner = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, -1.0),
    );
    let te = corner[vi];
    let seg = segs[iid];
    var out: VSOut;
    var sClip = uni.viewProj * vec4<f32>(seg.a, 1.0);
    var eClip = uni.viewProj * vec4<f32>(seg.b, 1.0);
    let nearW = 1e-5;
    if (sClip.w < nearW && eClip.w < nearW) {
        out.pos = vec4<f32>(0.0, 0.0, -1.0, 1.0); // both behind — clip offscreen
        out.rgba = vec4<f32>(0.0);
        out.edgeDist = 0.0;
        out.halfPx = 0.0;
        return out;
    }
    if (sClip.w < nearW) { let k = (nearW - sClip.w) / (eClip.w - sClip.w); sClip = mix(sClip, eClip, k); }
    else if (eClip.w < nearW) { let k = (nearW - eClip.w) / (sClip.w - eClip.w); eClip = mix(eClip, sClip, k); }
    let sNdc = sClip.xy / sClip.w;
    let eNdc = eClip.xy / eClip.w;
    let res = uni.res.xy;
    let dirPx = (eNdc - sNdc) * res;
    let lenPx = length(dirPx);
    let dir = select(vec2<f32>(1.0, 0.0), dirPx / lenPx, lenPx > 1e-4);
    let perp = vec2<f32>(-dir.y, dir.x);
    let halfW = max(seg.w, 1.0) * 0.5;
    let total = halfW + 1.0;
    let baseNdc = select(sNdc, eNdc, te.x > 0.5);
    let baseClip = select(sClip, eClip, te.x > 0.5);
    let ndc = baseNdc + perp * (te.y * total) * 2.0 / res;
    // real NDC depth so the rasterizer near-plane-clips a segment crossing in front of the camera (z=0
    // would skip that clip and let a near endpoint project to a huge NDC — the line "explodes"). No depth
    // attachment on this pass, so depth only clips; the gizmo still draws on top.
    out.pos = vec4<f32>(ndc, baseClip.z / baseClip.w, 1.0);
    out.rgba = unpack4x8unorm(seg.color);
    out.edgeDist = te.y * total;
    out.halfPx = halfW;
    return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
    let aa = 1.0 - smoothstep(in.halfPx - fwidth(in.edgeDist), in.halfPx + fwidth(in.edgeDist), abs(in.edgeDist));
    return vec4<f32>(in.rgba.rgb, in.rgba.a * aa);
}
`;

const _handleLines: {
    pipeline: GPURenderPipeline | null;
    segs: GPUBuffer | null;
    uni: GPUBuffer | null;
    bind: GPUBindGroup | null;
} = { pipeline: null, segs: null, uni: null, bind: null };

// pack linear rgb + alpha (each 0..1) into an RGBA8 u32 the line VS reads with `unpack4x8unorm` — no sRGB
// conversion, the colors go straight into the HDR framebuffer like the triangle handles
function packLine(r: number, g: number, b: number, a: number): number {
    const c = (x: number) => Math.max(0, Math.min(255, Math.round(x * 255)));
    return (c(r) | (c(g) << 8) | (c(b) << 16) | (c(a) << 24)) >>> 0;
}

function lift(c: number, t: number): number {
    return c + (1 - c) * t;
}

function smoothstep(e0: number, e1: number, x: number): number {
    const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
}

// append a screen-space triangle list (flat [x0,y0, x1,y1, …] corner pairs) to _handleData as NDC verts
// with a flat rgba; returns the next write offset
function pushQuad(
    off: number,
    corners: number[],
    width: number,
    height: number,
    r: number,
    g: number,
    b: number,
    a: number,
): number {
    for (let i = 0; i < corners.length; i += 2) {
        _handleData[off++] = (corners[i] / width) * 2 - 1;
        _handleData[off++] = 1 - (corners[i + 1] / height) * 2;
        _handleData[off++] = r;
        _handleData[off++] = g;
        _handleData[off++] = b;
        _handleData[off++] = a;
    }
    return off;
}

// a triangular arrowhead at the tip of an axis bar, pointing along it
function pushArrow(
    off: number,
    ox: number,
    oy: number,
    ex: number,
    ey: number,
    width: number,
    height: number,
    r: number,
    g: number,
    b: number,
    a: number,
): number {
    let dx = ex - ox;
    let dy = ey - oy;
    const l = Math.hypot(dx, dy) || 1;
    dx /= l;
    dy /= l;
    const px = -dy * ARROW_HALF;
    const py = dx * ARROW_HALF;
    const corners = [ex + dx * ARROW_LEN, ey + dy * ARROW_LEN, ex - px, ey - py, ex + px, ey + py];
    return pushQuad(off, corners, width, height, r, g, b, a);
}

// one screen-space square (centred at cx,cy, half-extent `h` px) — the scale tip cap / center point
function pushBox(
    off: number,
    cx: number,
    cy: number,
    h: number,
    width: number,
    height: number,
    r: number,
    g: number,
    b: number,
    a: number,
): number {
    const corners = [
        cx - h,
        cy - h,
        cx + h,
        cy - h,
        cx - h,
        cy + h,
        cx - h,
        cy + h,
        cx + h,
        cy - h,
        cx + h,
        cy + h,
    ];
    return pushQuad(off, corners, width, height, r, g, b, a);
}

// a filled circle (triangle fan) at a screen centre — the free-rotate interior fill, bounding the disc you
// grab for arcball rotation, so what's drawn matches what's grabbable (the pick's disc fallback)
function pushDisc(
    off: number,
    cx: number,
    cy: number,
    radius: number,
    width: number,
    height: number,
    r: number,
    g: number,
    b: number,
    a: number,
): number {
    let o = off;
    let px = cx + radius;
    let py = cy;
    for (let i = 1; i <= DISC_SEGS; i++) {
        const t = (i / DISC_SEGS) * Math.PI * 2;
        const x = cx + radius * Math.cos(t);
        const y = cy + radius * Math.sin(t);
        o = pushQuad(o, [cx, cy, px, py, x, y], width, height, r, g, b, a);
        px = x;
        py = y;
    }
    return o;
}

function drawHandles(eid: number, view: View): void {
    const { device } = Compute;
    const encoder = Render.encoder;
    if (!device || !encoder || !_handles.pipeline || !_handles.buffer) return;
    if (!view.framebuffer || view.width === 0 || view.height === 0) return;

    const origin = Handles.getOrigin();
    const manip = Handles.getManip();
    if (!origin || !manip) return;

    // the gizmo's world size — App computes it from the canvas CSS height so render matches the CSS-px pick
    const scale = Handles.getScale();
    if (scale <= 0) return;

    computeViewProj(eid, view.width / view.height, _handleVP);
    invert(_handleVP, _handleInv);
    // camera-forward in world: the ray through the viewport centre (for plane cull + screen/uniform drags)
    const eye = cursorRay(_handleInv, view.width / 2, view.height / 2, view.width, view.height).dir;
    const cap = manip === Scale ? "box" : "arrow";
    const axesNow = Handles.getAxes();
    const gs = glyphs(
        manip.handles,
        origin,
        axesNow,
        _handleVP,
        view.width,
        view.height,
        scale,
        eye,
        cap,
    );

    // a plane (id 3–5) fades to nothing as it turns edge-on, so it never pops; full opacity once it faces
    // the camera. Non-plane handles are unaffected (factor 1).
    const planeFade = (id: number): number => {
        if (id < 3 || id > 5) return 1;
        const n = axesNow[id - 3];
        return smoothstep(
            PLANE_EDGE_MIN,
            PLANE_EDGE_FADE,
            Math.abs(n[0] * eye[0] + n[1] * eye[1] + n[2] * eye[2]),
        );
    };

    if (_handlePalette !== current.palette) {
        _handlePalette = current.palette;
        _handleRGB.set(rgb(current.palette.axisX), 0);
        _handleRGB.set(rgb(current.palette.axisY), 3);
        _handleRGB.set(rgb(current.palette.axisZ), 6);
        _handleNeutral.set(rgb(current.palette.text), 0);
        _handleGrid.set(rgb(current.palette.grid), 0);
    }

    const w = view.width;
    const h = view.height;

    // a handle's color plus its state factor t (0 idle / 0.2 hover / 0.4 active), which also drives line
    // width + fill alpha
    const colorOf = (id: number): [number, number, number, number] => {
        const ax = decodeHandle(id).axis;
        const t = id === Handles.active ? 0.4 : id === Handles.hover ? 0.2 : 0;
        // axis handles brighten toward white on hover/active. The neutral handles (centre cube, screen
        // dot, trackball) are already ~white, so the same lift is invisible — instead dim them at idle and
        // brighten to full on state, so they read as secondary at rest yet still light up under the cursor.
        if (ax < 0) {
            const k = t >= 0.4 ? 1 : t >= 0.2 ? 0.82 : 0.6;
            return [_handleNeutral[0] * k, _handleNeutral[1] * k, _handleNeutral[2] * k, t];
        }
        return [
            lift(_handleRGB[ax * 3], t),
            lift(_handleRGB[ax * 3 + 1], t),
            lift(_handleRGB[ax * 3 + 2], t),
            t,
        ];
    };

    // the linear handles → world-space lines (axis shafts, rings, plane-square edges)
    const lines = handleSegments(manip.handles, origin, axesNow, scale, eye);
    let s = 0;
    for (const seg of lines) {
        if (s >= MAX_HANDLE_SEGS) break;
        const track = decodeHandle(seg.id).kind === "trackball";
        let [r, g, b, t] = colorOf(seg.id);
        // the outer roll ring is a faint secondary guide — grid-line color, lifting under the cursor
        if (track) {
            r = lift(_handleGrid[0], t);
            g = lift(_handleGrid[1], t);
            b = lift(_handleGrid[2], t);
        }
        const o = s * 8;
        _segData[o] = seg.a[0];
        _segData[o + 1] = seg.a[1];
        _segData[o + 2] = seg.a[2];
        _segData[o + 3] = t > 0 ? (track ? TRACK_W_ON : LINE_W_ON) : track ? TRACK_W : LINE_W;
        _segData[o + 4] = seg.b[0];
        _segData[o + 5] = seg.b[1];
        _segData[o + 6] = seg.b[2];
        // secondary roll ring sits more transparent than the primary axis rings (a bit firmer under cursor)
        const alpha = track ? TRACK_ALPHA + t : IDLE_ALPHA * planeFade(seg.id);
        _segU32[o + 7] = packLine(r, g, b, alpha);
        s++;
    }

    // the solid glyphs → screen triangles (arrowheads, scale caps, plane fill, centre dot); the shafts /
    // rings / plane edges are the lines above
    let off = 0;
    for (const g of gs) {
        if (!g) continue;
        const [r, gn, b, t] = colorOf(g.id);
        if (g.kind === "axis") {
            if (g.cap === "arrow")
                off = pushArrow(off, g.ox, g.oy, g.ex, g.ey, w, h, r, gn, b, IDLE_ALPHA);
            else if (g.cap === "box")
                off = pushBox(off, g.ex, g.ey, HANDLE_CAP, w, h, r, gn, b, IDLE_ALPHA);
        } else if (g.kind === "quad") {
            // a faint fill marks the grabbable square; its edges are the lines above. Triangulate the 4
            // corners into two triangles (6 verts) — pushQuad is a triangle-list, so feeding it 4 corners
            // would stitch a stray triangle across to the next glyph's corners
            const a = (t > 0 ? Math.min(1, PLANE_ALPHA + 0.4) : PLANE_ALPHA) * planeFade(g.id);
            const p = g.pts;
            const fill = [p[0], p[1], p[2], p[3], p[4], p[5], p[0], p[1], p[4], p[5], p[6], p[7]];
            off = pushQuad(off, fill, w, h, r, gn, b, a);
        } else if (g.kind === "point") {
            // the scale centre handle: a transparent white square (not a solid cube), firmer under the cursor
            const a = t > 0 ? 0.55 : 0.3;
            off = pushBox(
                off,
                g.cx,
                g.cy,
                CUBE_PX,
                w,
                h,
                _handleNeutral[0],
                _handleNeutral[1],
                _handleNeutral[2],
                a,
            );
        } else if (g.kind === "disc") {
            // the free-rotate interior: no fill at rest, a faint fill only under the cursor
            if (t > 0)
                off = pushDisc(
                    off,
                    g.cx,
                    g.cy,
                    g.r,
                    w,
                    h,
                    _handleGrid[0],
                    _handleGrid[1],
                    _handleGrid[2],
                    DISC_ALPHA,
                );
        }
    }

    if (s === 0 && off === 0) return;
    if (s > 0 && _handleLines.uni && _handleLines.segs) {
        _lineUni.set(_handleVP, 0);
        _lineUni[16] = view.width;
        _lineUni[17] = view.height;
        device.queue.writeBuffer(_handleLines.uni, 0, _lineUni);
        device.queue.writeBuffer(_handleLines.segs, 0, _segData, 0, s * 8);
    }
    if (off > 0) device.queue.writeBuffer(_handles.buffer, 0, _handleData, 0, off);

    const pass = encoder.beginRenderPass({
        label: "editor-handles",
        colorAttachments: [{ view: view.framebuffer, loadOp: "load", storeOp: "store" }],
        timestampWrites: Compute.span?.("editor:handles"),
    });
    if (s > 0 && _handleLines.pipeline && _handleLines.bind) {
        pass.setPipeline(_handleLines.pipeline);
        pass.setBindGroup(0, _handleLines.bind);
        pass.draw(6, s);
    }
    if (off > 0) {
        pass.setPipeline(_handles.pipeline);
        pass.setVertexBuffer(0, _handles.buffer);
        pass.draw(off / 6);
    }
    pass.end();
}

const HandleSystem: System = {
    name: "editor-handles",
    group: "draw",
    annotations: { mode: "edit", layer: "tooling" },
    after: [ColorSystem],
    before: [GlazeSystem],
    update(state: State) {
        // the editor camera — edit-only, carries Tag (the same camera PickSystem reads)
        for (const eid of state.query([Camera, Sear, Tag])) {
            const view = Views.get(eid);
            if (view?.framebuffer) {
                drawHandles(eid, view);
                break;
            }
        }
    },
};

async function warmHandles(device: GPUDevice): Promise<void> {
    const module = device.createShaderModule({ label: "editor-handles", code: HANDLE_SHADER });
    _handles.buffer?.destroy();
    _handles.buffer = device.createBuffer({
        label: "editor-handles",
        size: HANDLE_FLOATS * 4,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    _handles.pipeline = await device.createRenderPipelineAsync({
        label: "editor-handles",
        layout: "auto",
        vertex: {
            module,
            entryPoint: "vs",
            buffers: [
                {
                    arrayStride: 24,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x2" },
                        { shaderLocation: 1, offset: 8, format: "float32x4" },
                    ],
                },
            ],
        },
        fragment: {
            module,
            entryPoint: "fs",
            targets: [{ format: Render.format, blend: ALPHA_BLEND }],
        },
        primitive: { topology: "triangle-list" },
    });

    const lineModule = device.createShaderModule({
        label: "editor-handle-lines",
        code: HANDLE_LINE_SHADER,
    });
    _handleLines.segs?.destroy();
    _handleLines.uni?.destroy();
    _handleLines.segs = device.createBuffer({
        label: "editor-handle-lines",
        size: MAX_HANDLE_SEGS * 8 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _handleLines.uni = device.createBuffer({
        label: "editor-handle-lines-uni",
        size: _lineUni.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _handleLines.pipeline = await device.createRenderPipelineAsync({
        label: "editor-handle-lines",
        layout: "auto",
        vertex: { module: lineModule, entryPoint: "vs" },
        fragment: {
            module: lineModule,
            entryPoint: "fs",
            targets: [{ format: Render.format, blend: ALPHA_BLEND }],
        },
        primitive: { topology: "triangle-list" },
    });
    _handleLines.bind = device.createBindGroup({
        label: "editor-handle-lines",
        layout: _handleLines.pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: _handleLines.uni } },
            { binding: 1, resource: { buffer: _handleLines.segs } },
        ],
    });
}

function disposeHandles(): void {
    _handles.buffer?.destroy();
    _handles.buffer = null;
    _handles.pipeline = null;
    _handleLines.segs?.destroy();
    _handleLines.uni?.destroy();
    _handleLines.segs = null;
    _handleLines.uni = null;
    _handleLines.pipeline = null;
    _handleLines.bind = null;
    Handles.getOrigin = () => null;
    Handles.getAxes = () => WORLD_AXES;
    Handles.getManip = () => null;
    Handles.getScale = () => 0;
    Handles.active = -1;
    Handles.hover = -1;
}

/** the editor's viewport chrome as one tooling plugin: the ground grid, the selection outline, and the
 * transform gizmo. Injected into the editor host (see `plugins.ts` `TOOLING_PLUGINS`); a shipped game
 * never imports it, so it tree-shakes out. */
export const GizmosPlugin: Plugin = {
    name: "Gizmos",
    components: { Overlays },
    traits: { Overlays: { defaults: () => ({ enabled: Overlay.Grid | Overlay.Outline }) } },
    systems: [GridSystem, OutlineSystem, HandleSystem],
    dependencies: [RenderPlugin],
    async warm() {
        const { device } = Compute;
        if (!device) return;
        await Promise.all([warmGrid(device), warmOutline(device), warmHandles(device)]);
    },
    dispose() {
        disposeGrid();
        disposeOutline();
        disposeHandles();
    },
};
