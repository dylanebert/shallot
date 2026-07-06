// Editor viewport picking — clicking the scene selects the entity under the cursor. Like the grid +
// outline overlay (`viewport.ts`), it's an editor concern layered on a neutral engine primitive: sear's
// prepass id lane (`view.tag`, the camera carries `Tag`). It reads that lane back through `Mirror` (no GPU
// stall, the same pattern orrstead's hover uses), so picking is pixel-perfect with no CPU raycast and the
// selection outline that already samples `view.tag` lights up for free. Tooling-layer + edit-only: the
// readback acts on the editor camera, tree-shakes out of a shipped game, and never runs in play mode (the
// game brings its own picking if it wants one). The host (App.svelte) reads {@link Pick} on a click and
// routes it to `doc.select` — the eid → node resolution + the click-vs-drag gate are the pure helpers here.
import {
    Camera,
    Compute,
    InputPlugin,
    Inputs,
    type Mirror,
    MirrorPlugin,
    mirror,
    type Plugin,
    RenderPlugin,
    Sear,
    type System,
    Tag,
} from "@dylanebert/shallot";
import type { Node } from "@dylanebert/shallot/editor";
import { Render, Views } from "@dylanebert/shallot/render/core";
import { ColorSystem, TAG_NONE } from "@dylanebert/shallot/sear/core";

/**
 * the entity under the cursor this frame, read back from the editor camera's prepass id lane
 * (`view.tag`). `-1` over empty space or off-canvas. The readback is 1-2 frames stale (no GPU stall, like
 * orrstead's hover) — fine for a click, where the cursor is still. The host reads this on a viewport click
 * and resolves it to a scene node ({@link nodeForEid}) to drive `doc.select`.
 */
export const Pick = { eid: -1 };

/** a press→release that moved less than this many CSS px is a select-click; more is an orbit drag. */
export const CLICK_SLOP = 4;

/**
 * classify a viewport press→release: a click (small movement → select the entity under the cursor) versus
 * a drag (the orbit camera moved → ignore). Squared-distance against {@link CLICK_SLOP} so a tap that
 * jitters a pixel still selects while an orbit drag never does.
 */
export function isClick(
    downX: number,
    downY: number,
    upX: number,
    upY: number,
    slop = CLICK_SLOP,
): boolean {
    const dx = upX - downX;
    const dy = upY - downY;
    return dx * dx + dy * dy <= slop * slop;
}

/**
 * the selection after a viewport / outliner pick, the toggle-everywhere scheme: a plain pick selects
 * only `picked` (or clears on empty space, `null`); a modifier pick (`additive`) toggles `picked` in or
 * out of the set, and is a no-op on empty space so a stray shift-click never drops the selection. Pure —
 * the caller diffs it against `doc.selection`. Insertion order is preserved, so the last entry is the
 * active (last-picked) node a pivot / local frame reads.
 */
export function nextSelection(current: Node[], picked: Node | null, additive: boolean): Node[] {
    if (!additive) return picked ? [picked] : [];
    if (!picked) return current;
    return current.includes(picked) ? current.filter((n) => n !== picked) : [...current, picked];
}

/**
 * the scene node a picked eid selects: the node whose eid matches in the load-time `nodeMap`, or `null`
 * for empty space (`eid < 0`) or an eid no node owns (a custom-surface tag, a since-despawned entity) —
 * the caller deselects on `null`. Linear over the map; clicks are rare and scenes small.
 */
export function nodeForEid(eid: number, nodeMap: Map<Node, number>): Node | null {
    if (eid < 0) return null;
    for (const [node, e] of nodeMap) if (e === eid) return node;
    return null;
}

// copyTextureToBuffer wants a 256-byte-aligned row; the one r32uint cursor texel rides the first 4 bytes
const READBACK_BYTES = 256;

let _buf: GPUBuffer | null = null;
let _readback: Mirror | null = null;
// Compute.frame when the current uninterrupted on-canvas hover began. A snapshot captured before it (a
// pixel from before the cursor last left + re-entered the canvas) is rejected, not applied, so a click
// right after re-entry never reads a stale entity.
let _hoverSince = Number.POSITIVE_INFINITY;

// after sear's color pass wrote the framebuffer + the prepass `view.tag`: decode the latest readback into
// Pick.eid, then copy this frame's cursor pixel for a later frame to read (1-2 frames of GPU lag, no
// stall). The editor's entities are instanced Parts, so a tag is the eid directly (TAG_NONE = no surface).
const PickSystem: System = {
    name: "editor-pick",
    group: "draw",
    annotations: { mode: "edit", layer: "tooling" },
    after: [ColorSystem],
    update(state) {
        if (!Render.encoder || !_buf || !_readback) return;

        // the editor camera: the on-screen Sear camera carrying the Tag lane. In edit mode only it carries
        // Tag (syncCameraEffects excludes Tag from the scene camera), so this resolves it uniquely.
        let tagTex: GPUTexture | null = null;
        let width = 0;
        let height = 0;
        for (const eid of state.query([Camera, Sear, Tag])) {
            const view = Views.get(eid);
            if (view?.tag && view.canvas) {
                tagTex = view.tag;
                width = view.width;
                height = view.height;
                break;
            }
        }

        const m = Inputs.mouse;
        if (!tagTex || !m.hover || m.canvasWidth <= 0 || m.canvasHeight <= 0) {
            Pick.eid = -1;
            _hoverSince = Number.POSITIVE_INFINITY;
            return;
        }
        if (_hoverSince === Number.POSITIVE_INFINITY) _hoverSince = Compute.frame;

        // apply the latest readback only if its pixel was sampled during this hover session — the frame
        // stamp rejects a pixel left over from before the cursor last left the canvas
        const snap = _readback.snapshot;
        const tag = snap && snap.frame >= _hoverSince ? new Uint32Array(snap.bytes)[0] : TAG_NONE;
        Pick.eid = tag === TAG_NONE ? -1 : tag;

        // copy the cursor pixel for a future frame's readback. mouse x/y are CSS px; scale to the tag
        // texture's device px
        const px = Math.min(width - 1, Math.floor((m.x / m.canvasWidth) * width));
        const py = Math.min(height - 1, Math.floor((m.y / m.canvasHeight) * height));
        if (px < 0 || py < 0) return;
        Render.encoder.copyTextureToBuffer(
            { texture: tagTex, origin: { x: px, y: py } },
            { buffer: _buf, bytesPerRow: READBACK_BYTES, rowsPerImage: 1 },
            { width: 1, height: 1 },
        );
    },
};

/**
 * cursor → {@link Pick} for the editor viewport. Reads sear's `view.tag` at the cursor pixel through
 * `Mirror` (three staging buffers in flight, so no `mapAsync` stall). Editor tooling: its system is
 * `layer: "tooling"` (stripped from a shipped game) and `mode: "edit"` (never runs in play mode). The host
 * spreads it into the editor build via `TOOLING_PLUGINS`; the editor camera carries `Tag` so `view.tag`
 * is filled. The engine owns the tag target; this owns the readback + decode, the host owns the select.
 */
export const PickPlugin: Plugin = {
    name: "Pick",
    dependencies: [InputPlugin, RenderPlugin, MirrorPlugin],
    systems: [PickSystem],

    warm() {
        _buf?.destroy();
        _buf = Compute.device.createBuffer({
            label: "editor-pick-readback",
            size: READBACK_BYTES,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        _readback = mirror(_buf, { ring: 3 });
        _hoverSince = Number.POSITIVE_INFINITY;
        Pick.eid = -1;
    },

    dispose() {
        _readback?.dispose();
        _buf?.destroy();
        _readback = null;
        _buf = null;
        Pick.eid = -1;
    },
};
