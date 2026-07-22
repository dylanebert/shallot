// The per-sample overlay channel for the tumble sample host: the demonstration layer a tumble.js sample's
// `render(draw)` override draws — threshold labels, load/overlap HUD readouts, cast rays, impact markers,
// query-box outlines. The engine fold's first pass reproduced each sample's `build()` + `update()` but
// dropped `render()`, so event/collision scenarios read as bare "boxes falling"; this restores it.
//
// The channel is VISUAL-ONLY, derived entirely from the live world each frame (never a hand-authored
// position — the anti-divergence invariant): the host runs `config.render` after stepping, against the same
// world the gold verified, so nothing here can drift from the physics and nothing here feeds the oracle.
//
// The DrawApi surface mirrors tumble.js's `samples/src/gfx/draw.ts`, trimmed to what the corpus actually
// uses (`cube`/`arrow`/`cross`/`axes` are unused — no adapter). Geometric primitives (line/point/aabb + the
// wireframe solid* pair) emit through the same `Lines` idiom the host draws joints/contacts with; `string3d`
// projects to positioned HTML labels over the canvas (`tumble-project.ts`); `text` (the sample's `drawText`)
// appends to the gym `#hud` readout.

import { unpackColor } from "@dylanebert/shallot";
import { segment } from "@dylanebert/shallot/extras";
import { qRotate } from "@dylanebert/shallot/physics/core";
import type { Capsule, Sphere, Vec3, World, WorldTransform } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";
import { cameraPose, worldToScreen } from "./tumble-project";

/** the immediate-draw surface a sample's `render()` calls — the corpus subset of tumble.js's `DrawApi`, plus
 *  `text` (the sample base's `drawText`). Colors are low-24-bit packed RGB, matching the source. */
export interface Overlay {
    /** line segment `a`→`b`. */
    line(a: Vec3, b: Vec3, color: number): void;
    /** point marker at `p` (`size` in pixels, mapped to a small world cross). */
    point(p: Vec3, size: number, color: number): void;
    /** wireframe axis-aligned box between world `min`/`max`. */
    aabb(min: Vec3, max: Vec3, color: number): void;
    /** wireframe sphere; `transform`'s rotation places `sphere.center` in the world. */
    solidSphere(transform: WorldTransform, sphere: Sphere, color: number): void;
    /** wireframe capsule along its two local centers, placed by `transform`. */
    solidCapsule(transform: WorldTransform, capsule: Capsule, color: number): void;
    /** positioned HTML label at world point `p`. */
    string3d(p: Vec3, s: string, color: number): void;
    /** append one HUD readout line (the sample base's `drawText`). */
    text(line: string): void;
}

/** a ported sample's `render()` — draws its overlay against the live `world` each frame. Runs only in the
 *  live view (never the oracle replay), reads the world but must not mutate it, and derives every position
 *  from world/sample state (no hand-authored coordinates). `stepCount` mirrors the sample's `this.stepCount`
 *  for the time-driven sweeps (the ray curtain, the overlap box). */
export type SampleRender = (
    draw: Overlay,
    world: World,
    params: SampleParams,
    stepCount: number,
) => void;

interface Label {
    x: number;
    y: number;
    z: number;
    s: string;
    color: number;
}

/** the host-owned overlay handle: the {@link Overlay} the render hook draws into, a begin/end bracket the
 *  draw system runs it between, and the readouts `live()` (HUD lines) + the probe (label/text counts) read. */
export interface OverlayLayer {
    api: Overlay;
    /** clear the frame's collected labels + HUD lines before the render hook runs. */
    begin(): void;
    /** project the collected labels to positioned HTML + commit the HUD lines (read by `hudLines`). */
    end(): void;
    /** this frame's committed HUD readout lines (the sample's `drawText` output). */
    hudLines(): string[];
    /** the count of `string3d` labels the last render pass emitted — the adapter's own output, before
     *  projection. The headless wiring test asserts this (no sized canvas there, so nothing projects). */
    labelCount(): number;
    /** the count of labels the last render pass actually projected on-screen (post front/bounds cull) —
     *  the positive presence signal the visual probe asserts nonzero on a real device. */
    shownCount(): number;
    /** the count of HUD lines the last render pass emitted. */
    textCount(): number;
    /** remove the label container (registered on `state.onDispose`). */
    dispose(): void;
}

// --- wireframe solids (the Lines idiom) -------------------------------------------------------------------

// one world-space circle: `segs` chords around `center` in the plane spanned by unit `u`, `v`.
function ring(
    cx: number,
    cy: number,
    cz: number,
    ux: number,
    uy: number,
    uz: number,
    vx: number,
    vy: number,
    vz: number,
    r: number,
    segs: number,
    color: number,
): void {
    let px = 0;
    let py = 0;
    let pz = 0;
    for (let i = 0; i <= segs; i++) {
        const a = (2 * Math.PI * i) / segs;
        const ca = Math.cos(a) * r;
        const sa = Math.sin(a) * r;
        const x = cx + ca * ux + sa * vx;
        const y = cy + ca * uy + sa * vy;
        const z = cz + ca * uz + sa * vz;
        if (i > 0) segment([px, py, pz], [x, y, z], color);
        px = x;
        py = y;
        pz = z;
    }
}

// wireframe sphere as three orthogonal great circles.
function wireSphere(cx: number, cy: number, cz: number, r: number, color: number): void {
    ring(cx, cy, cz, 1, 0, 0, 0, 1, 0, r, 20, color);
    ring(cx, cy, cz, 0, 1, 0, 0, 0, 1, r, 20, color);
    ring(cx, cy, cz, 1, 0, 0, 0, 0, 1, r, 20, color);
}

function norm(x: number, y: number, z: number): [number, number, number] {
    const l = Math.hypot(x, y, z) || 1;
    return [x / l, y / l, z / l];
}

// wireframe capsule: an equator ring at each hemisphere center plus `M` meridian silhouettes (each a curve
// from bottom pole, down the tube, to top pole), matching `tumble-solids.ts`'s solid-mesh parametrization.
const CAP_MERIDIANS = 6;
const CAP_ARC = 5;
function wireCapsule(c1: Vec3, c2: Vec3, r: number, color: number): void {
    const [ax, ay, az] = norm(c2.x - c1.x, c2.y - c1.y, c2.z - c1.z);
    let [ux, uy, uz] = norm(-ay, ax, 0);
    if (ux * ux + uy * uy + uz * uz < 0.5) [ux, uy, uz] = norm(0, -az, ay);
    const wx = ay * uz - az * uy;
    const wy = az * ux - ax * uz;
    const wz = ax * uy - ay * ux;

    ring(c1.x, c1.y, c1.z, ux, uy, uz, wx, wy, wz, r, 20, color);
    ring(c2.x, c2.y, c2.z, ux, uy, uz, wx, wy, wz, r, 20, color);

    // each meridian: bottom hemisphere (cap at c1, axis −a) then top hemisphere (cap at c2, axis +a).
    for (let m = 0; m < CAP_MERIDIANS; m++) {
        const theta = (2 * Math.PI * m) / CAP_MERIDIANS;
        const dx = Math.cos(theta) * ux + Math.sin(theta) * wx;
        const dy = Math.cos(theta) * uy + Math.sin(theta) * wy;
        const dz = Math.cos(theta) * uz + Math.sin(theta) * wz;
        let px = 0;
        let py = 0;
        let pz = 0;
        let first = true;
        const arc = (c: Vec3, sign: number, k: number): void => {
            const beta = (Math.PI / 2) * k;
            const sb = Math.sin(beta);
            const cb = Math.cos(beta);
            const x = c.x + r * (sign * sb * ax + cb * dx);
            const y = c.y + r * (sign * sb * ay + cb * dy);
            const z = c.z + r * (sign * sb * az + cb * dz);
            if (!first) segment([px, py, pz], [x, y, z], color);
            px = x;
            py = y;
            pz = z;
            first = false;
        };
        for (let k = 0; k <= CAP_ARC; k++) arc(c1, -1, 1 - k / CAP_ARC);
        for (let k = 0; k <= CAP_ARC; k++) arc(c2, 1, k / CAP_ARC);
    }
}

// world center of a solidSphere/solidCapsule local point under `xf` (matches tumble.js `pushSphere`).
function place(xf: WorldTransform, c: Vec3): [number, number, number] {
    const q = xf.q;
    const [rx, ry, rz] = qRotate(q.v.x, q.v.y, q.v.z, q.s, c.x, c.y, c.z);
    return [xf.p.x + rx, xf.p.y + ry, xf.p.z + rz];
}

// --- the overlay layer ------------------------------------------------------------------------------------

/**
 * Build the overlay layer over `canvas` for the camera entity `cam`. Creates a pointer-transparent label
 * container above the canvas (removed by {@link OverlayLayer.dispose}, wired to `state.onDispose` by the
 * host). The returned {@link Overlay} emits geometry through `Lines` immediately and buffers labels + HUD
 * lines until {@link OverlayLayer.end} projects/commits them.
 */
export function overlayLayer(cam: number, canvas: HTMLCanvasElement): OverlayLayer {
    const host = document.createElement("div");
    host.style.cssText =
        "position:absolute;inset:0;overflow:hidden;pointer-events:none;font-family:'JetBrains Mono',ui-monospace,monospace;";
    (canvas.parentElement ?? document.body).appendChild(host);

    const pool: HTMLDivElement[] = [];
    const labels: Label[] = [];
    const texts: string[] = [];
    let committed: string[] = [];
    let lastLabels = 0;
    let lastShown = 0;

    const point = (p: Vec3, size: number, color: number): void => {
        const r = 0.02 * Math.max(size, 1);
        segment([p.x - r, p.y, p.z], [p.x + r, p.y, p.z], color, 3);
        segment([p.x, p.y - r, p.z], [p.x, p.y + r, p.z], color, 3);
        segment([p.x, p.y, p.z - r], [p.x, p.y, p.z + r], color, 3);
    };

    const api: Overlay = {
        line(a, b, color) {
            segment([a.x, a.y, a.z], [b.x, b.y, b.z], color);
        },
        point,
        aabb(min, max, color) {
            const x0 = min.x;
            const y0 = min.y;
            const z0 = min.z;
            const x1 = max.x;
            const y1 = max.y;
            const z1 = max.z;
            segment([x0, y0, z0], [x1, y0, z0], color);
            segment([x1, y0, z0], [x1, y0, z1], color);
            segment([x1, y0, z1], [x0, y0, z1], color);
            segment([x0, y0, z1], [x0, y0, z0], color);
            segment([x0, y1, z0], [x1, y1, z0], color);
            segment([x1, y1, z0], [x1, y1, z1], color);
            segment([x1, y1, z1], [x0, y1, z1], color);
            segment([x0, y1, z1], [x0, y1, z0], color);
            segment([x0, y0, z0], [x0, y1, z0], color);
            segment([x1, y0, z0], [x1, y1, z0], color);
            segment([x1, y0, z1], [x1, y1, z1], color);
            segment([x0, y0, z1], [x0, y1, z1], color);
        },
        solidSphere(xf, sphere, color) {
            const [cx, cy, cz] = place(xf, sphere.center);
            wireSphere(cx, cy, cz, sphere.radius, color);
        },
        solidCapsule(xf, cap, color) {
            const [c1x, c1y, c1z] = place(xf, cap.center1);
            const [c2x, c2y, c2z] = place(xf, cap.center2);
            wireCapsule({ x: c1x, y: c1y, z: c1z }, { x: c2x, y: c2y, z: c2z }, cap.radius, color);
        },
        string3d(p, s, color) {
            labels.push({ x: p.x, y: p.y, z: p.z, s, color });
        },
        text(line) {
            texts.push(line);
        },
    };

    return {
        api,
        begin() {
            labels.length = 0;
            texts.length = 0;
        },
        end() {
            lastLabels = labels.length;
            committed = texts.slice();

            const rect = canvas.getBoundingClientRect();
            const w = rect.width;
            const h = rect.height;
            const pose = cameraPose(cam);
            let shown = 0;
            if (w >= 1 && h >= 1) {
                for (const l of labels) {
                    const s = worldToScreen(pose.pos, pose.quat, pose.fovDeg, w, h, l);
                    // negated inclusive bounds so a NaN projection is culled, not written as `left:NaNpx`
                    if (!s.front || !(s.x >= 0 && s.x <= w && s.y >= 0 && s.y <= h)) continue;
                    let el = pool[shown];
                    if (!el) {
                        el = document.createElement("div");
                        el.style.cssText =
                            "position:absolute;transform:translate(-50%,-50%);white-space:nowrap;font-size:11px;line-height:1;text-shadow:0 1px 2px rgba(0,0,0,0.9);pointer-events:none;";
                        host.appendChild(el);
                        pool.push(el);
                    }
                    const c = unpackColor(l.color);
                    el.textContent = l.s;
                    el.style.left = `${s.x}px`;
                    el.style.top = `${s.y}px`;
                    el.style.color = `rgb(${(c.r * 255) | 0},${(c.g * 255) | 0},${(c.b * 255) | 0})`;
                    el.style.display = "block";
                    shown++;
                }
            }
            for (let i = shown; i < pool.length; i++) pool[i].style.display = "none";
            lastShown = shown;
        },
        hudLines() {
            return committed;
        },
        labelCount() {
            return lastLabels;
        },
        shownCount() {
            return lastShown;
        },
        textCount() {
            return committed.length;
        },
        dispose() {
            host.remove();
        },
    };
}
