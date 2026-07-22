import {
    Camera,
    Inputs,
    mountOverlay,
    Orbit,
    type Plugin,
    type State,
    type System,
    Transform,
} from "@dylanebert/shallot";
import { segment } from "@dylanebert/shallot/extras";
import { cursorRay, qRotate } from "@dylanebert/shallot/physics/core";
import { brush, march } from "./voxel/edit";
import { generate } from "./voxel/generate";
import { commitEdit, syncGrid, Voxels } from "./voxel/mesher";

// Realtime carving (the voxel showcase's interactive layer). A top toolbar picks the pointer (orbit) or
// terrain (sculpt) tool (keys V / B). In the terrain tool left-drag adds, shift+drag carves, scroll resizes
// the brush (a camera-facing ring previews the footprint), and orbit moves to middle-drag. The grid is a
// DENSITY field, and the brush blends a falloff WEIGHT into it (Astroneer / Planet-Coaster deform): the
// cursor ray re-marches to the live surface each frame, blends `FLOW·dt` at the hit cell, and cells cross
// the ISO threshold gradually — so terrain grows continuously outward along its normal, not a hard sphere
// on a plane. Only the chunk slices whose occupancy flips re-upload + re-mesh that frame.

const DEFAULT_RADIUS = 8; // a large, scoop-sized default brush (scroll resizes it)
const MAX_RADIUS = 40;
// brush flow: density per second blended at the brush centre, at the DEFAULT_RADIUS reference (× dt →
// framerate-independent, × falloff → soft edge). Small, so the surface grows continuously across ISO over
// the stroke rather than a hard stamp. Normalized by √(DEFAULT_RADIUS/radius) per dab so a small brush
// (steep falloff) isn't sluggish and a large one (broad core) isn't runaway.
const FLOW = 5.0;
// the smoothstep falloff only crosses ISO within ~⅔ of `radius` in a normal hold, so the footprint ring is
// drawn at that effective fraction (matching what's visibly affected, not the full math extent).
const RING_FRAC = 0.66;
const REACH = 2000; // the cursor-ray cap, generous past the orbit's max distance; the march clips to the grid
const RING_SEGMENTS = 48; // the brush-footprint ring drawn on the surface

let tool: "pointer" | "terrain" = "pointer";
let radius = DEFAULT_RADIUS; // scroll-driven, shown by the ring on the pointer (not an editable side control)
let camEid = -1;
let canvasEl: HTMLCanvasElement | null = null;
let baseOrbitButton = 0;
let basePanButton = 1;
let baseZoomSpeed = 0.025;
let seed = 1337;
let appliedSeed = 1337; // the seed currently meshed; a mismatch with `seed` triggers a live re-gen

/** capture the orbit camera + canvas the carve tool aims through, and the authored orbit scheme (the pointer
 *  tool restores it, the terrain tool remaps left→brush / scroll→brush-size). Call once after the scene warms. */
export function initCarve(
    state: State,
    canvas: HTMLCanvasElement | null,
    initialSeed: number,
): void {
    tool = "pointer"; // a fresh run starts in the pointer tool (module state survives HMR otherwise)
    radius = DEFAULT_RADIUS;
    seed = initialSeed;
    appliedSeed = initialSeed; // dormant until a live reseed (F9 / setSeed)
    canvasEl = canvas;
    camEid = -1;
    for (const eid of state.query([Camera])) {
        camEid = eid;
        break;
    }
    if (camEid >= 0 && state.has(camEid, Orbit)) {
        baseOrbitButton = Orbit.orbitButton.get(camEid);
        basePanButton = Orbit.panButton.get(camEid);
        baseZoomSpeed = Orbit.zoomSpeed.get(camEid);
    }
}

/** reseed the generated terrain in place (the F9 reseed) — VoxelControlSystem regenerates on the next frame. */
export function setSeed(s: number): void {
    seed = s;
}

// live re-generation: poll `seed` each frame and redispatch the generator when it changes — the F9 reseed
// flows through here, so it needs no page reload. The generator pipeline is cached after the first call, so
// a re-gen is a perm-table rebuild + dispatch (no recompile).
const VoxelControlSystem: System = {
    name: "voxel-control",
    group: "simulation",
    update() {
        if (!Voxels.grid || seed === appliedSeed) return;
        appliedSeed = seed;
        // re-sync the CPU mirror after the GPU regen so a carve marches/edits the new terrain, not the old.
        void generate(seed).then(syncGrid);
    },
};

// the brush cursor: a camera-FACING ring (billboard) at the world point `c`, drawn in the camera's right/up
// plane so it never reorients per voxel face — persistent while you hover, only turning when you orbit. Cyan
// adds, rose subtracts. Pulled a touch toward the camera so it sits in front of the surface (lines are
// depth-tested, depth-write off).
function drawBrushRing(c: readonly [number, number, number], r: number, sub: boolean): void {
    const qx = Transform.rot.x.get(camEid);
    const qy = Transform.rot.y.get(camEid);
    const qz = Transform.rot.z.get(camEid);
    const qw = Transform.rot.w.get(camEid);
    const right = qRotate(qx, qy, qz, qw, 1, 0, 0);
    const up = qRotate(qx, qy, qz, qw, 0, 1, 0);
    const fwd = qRotate(qx, qy, qz, qw, 0, 0, 1); // camera -Z is view dir; +Z points back toward the camera
    const cx = c[0] + fwd[0] * 0.5;
    const cy = c[1] + fwd[1] * 0.5;
    const cz = c[2] + fwd[2] * 0.5;
    const color = sub ? 0xfb7185 : 0x67e8f9;
    let px = 0;
    let py = 0;
    let pz = 0;
    for (let i = 0; i <= RING_SEGMENTS; i++) {
        const a = (i / RING_SEGMENTS) * Math.PI * 2;
        const ca = Math.cos(a) * r;
        const sa = Math.sin(a) * r;
        const x = cx + right[0] * ca + up[0] * sa;
        const y = cy + right[1] * ca + up[1] * sa;
        const z = cz + right[2] * ca + up[2] * sa;
        if (i > 0) segment([px, py, pz], [x, y, z], color, 2);
        px = x;
        py = y;
        pz = z;
    }
}

// Two tools (toolbar / keys V · B): the pointer tool is the authored orbit scheme; the terrain tool frees
// the left button for the brush — orbit moves to middle-drag (`orbitButton = 1`), pan moves to right-drag so
// it clears the new middle-pan default, and zoom is silenced (`zoomSpeed = 0`) so scroll sizes the brush.
// Left-drag adds, Shift+drag carves. The brush ADDS WEIGHT to the density field (Astroneer / Planet-Coaster
// deform): each frame it re-marches the cursor ray to the live surface and blends a √-normalized `FLOW·dt`
// weight with a radial falloff, centred AT the surface, so the isosurface grows continuously across ISO
// outward along its normal — center-first, no hard stamp, no plane. The re-mesh fires only on the frames a
// cell actually crosses ISO. Runs after OrbitPlugin so the ray reads the freshly-posed camera.
const CarveSystem: System = {
    name: "voxel-carve",
    group: "simulation",
    update(state) {
        if (camEid < 0) return;
        const terrain = tool === "terrain";

        // remap the orbit controls to the active tool (idempotent; a 1-frame lag on switch is invisible).
        if (state.has(camEid, Orbit)) {
            Orbit.orbitButton.set(camEid, terrain ? 1 : baseOrbitButton);
            Orbit.panButton.set(camEid, terrain ? 2 : basePanButton);
            Orbit.zoomSpeed.set(camEid, terrain ? 0 : baseZoomSpeed);
        }

        if (!terrain || !Voxels.data) {
            if (canvasEl) canvasEl.style.cursor = "";
            return;
        }

        // scroll (zoom is off in this tool) resizes the brush.
        if (Inputs.mouse.scroll !== 0) {
            radius = Math.min(MAX_RADIUS, Math.max(1, radius - Math.sign(Inputs.mouse.scroll)));
        }

        const ray = cursorRay(state, camEid);
        const hit = ray ? march(Voxels.data, ray.origin, ray.dir, REACH) : null;
        const sub = Inputs.isKeyDown("ShiftLeft") || Inputs.isKeyDown("ShiftRight");

        // ring at the surface contact previews the footprint; the OS cursor hides so the ring is the pointer.
        if (canvasEl) canvasEl.style.cursor = hit ? "none" : "";
        if (hit && ray) {
            drawBrushRing(
                [
                    ray.origin[0] + ray.dir[0] * hit.distance,
                    ray.origin[1] + ray.dir[1] * hit.distance,
                    ray.origin[2] + ray.dir[2] * hit.distance,
                ],
                radius * RING_FRAC,
                sub,
            );
        }

        if (!Inputs.mouse.left || !hit) return;
        // centre the brush AT the surface: carve the first solid cell, add the air cell across the face. The
        // peak weight lands on the boundary, so geometry appears/erodes right at the surface (not buried
        // inward) and grows outward. Flow is √-normalized by radius so the surface advances at a comparable
        // rate across brush sizes; dt-scaled so it's framerate-independent.
        const target = sub ? hit.cell : hit.place;
        const flow = FLOW * Math.sqrt(DEFAULT_RADIUS / radius);
        const delta = (sub ? -1 : 1) * flow * state.time.deltaTime;
        const touched = brush(Voxels.data, target[0], target[1], target[2], radius, delta);
        if (touched.size > 0) commitEdit(touched);
    },
};

const VoxelControlPlugin: Plugin = {
    name: "VoxelControl",
    systems: [VoxelControlSystem, CarveSystem],
};

export default VoxelControlPlugin;

// Lucide icons (mouse-pointer-2, mountain) — the two-tool top toolbar.
const POINTER_ICON = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.037 4.688a.495.495 0 0 1 .651-.651l16 6.5a.5.5 0 0 1-.063.947l-6.124 1.58a2 2 0 0 0-1.438 1.435l-1.579 6.126a.5.5 0 0 1-.947.063z"/></svg>`;
const TERRAIN_ICON = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m8 3 4 8 5-5 5 15H2L8 3z"/></svg>`;

const TOOLBAR_CSS = `
.voxel-toolbar { position: absolute; top: 12px; left: 50%; transform: translateX(-50%); display: flex; gap: 4px; padding: 4px; pointer-events: auto; background: rgba(14, 13, 12, 0.72); border: 1px solid rgba(255, 255, 255, 0.09); border-radius: 10px; backdrop-filter: blur(8px); z-index: 20; }
.voxel-tool { display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; padding: 0; background: transparent; border: none; border-radius: 7px; color: #a09890; cursor: pointer; transition: background 120ms, color 120ms; }
.voxel-tool:hover { background: rgba(255, 255, 255, 0.06); color: #f0ece8; }
.voxel-tool.active { background: rgba(232, 168, 107, 0.16); color: #e8a86b; }
`;

// the modern top toolbar: two Lucide icon buttons (pointer / terrain), the active one warm-accented. Returns
// `setTool` (also driven by keys V / B) + a cleanup. The single source for the active tool is module `tool`.
export function mountToolbar(): {
    setTool: (t: "pointer" | "terrain") => void;
    dispose: () => void;
} {
    // the toolbar lives in the engine's sandboxed overlay (`mountOverlay`) — the canvas-bounded surface,
    // not `document.body` (a fixed bar there spills past the canvas, into the rest of the page).
    const overlay = mountOverlay(document.querySelector("canvas"));
    const style = document.createElement("style");
    style.textContent = TOOLBAR_CSS;
    document.head.appendChild(style);
    const bar = document.createElement("div");
    bar.className = "voxel-toolbar";

    const defs = [
        { id: "pointer" as const, title: "Pointer — orbit (V)", svg: POINTER_ICON },
        {
            id: "terrain" as const,
            title: "Terrain — click adds · shift subtracts · scroll resizes (B)",
            svg: TERRAIN_ICON,
        },
    ];
    const btns = new Map<string, HTMLButtonElement>();
    const setTool = (t: "pointer" | "terrain") => {
        tool = t;
        for (const [id, b] of btns) b.classList.toggle("active", id === t);
    };
    for (const d of defs) {
        const b = document.createElement("button");
        b.className = "voxel-tool";
        b.title = d.title;
        b.innerHTML = d.svg;
        b.onclick = () => setTool(d.id);
        btns.set(d.id, b);
        bar.appendChild(b);
    }
    overlay.appendChild(bar);
    setTool(tool);
    return {
        setTool,
        dispose: () => {
            overlay.remove();
            style.remove();
        },
    };
}
