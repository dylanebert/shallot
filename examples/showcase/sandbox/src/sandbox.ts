import {
    AmbientLight,
    Body,
    Camera,
    CameraMode,
    Character,
    Depth,
    Fog,
    Glaze,
    Inputs,
    Listener,
    type Mirror,
    mirror,
    mountOverlay,
    Physics,
    Player,
    type Plugin,
    RenderPlugin,
    Resolution,
    Sear,
    ShapeKind,
    type State,
    type System,
    Transform,
} from "@dylanebert/shallot";
import { Avbd, type JointDef } from "@dylanebert/shallot/avbd/core";
import { fsCtxSchema, lit, registerSurface, surfaceLayout } from "@dylanebert/shallot/sear/core";
import { unpackLdrColor, Xform } from "@dylanebert/shallot/utils/core";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { armImpacts, ImpactSystem, registerInstruments } from "./audio";
import { type Gun, gun } from "./gun";
import { Brick, box, brickStack, bridge, hex, lamp, pyramid, rope } from "./spawn";
import { hud, isTouchOnly, setCrosshair, touchNotice } from "./ui";

// The sandbox — the first-person gravity-gun showcase (physics + player + synthetic audio together).
// A manifest project: shallot.json enables physics + player + audio + this plugin and sets the
// load-bearing capacity, the empty scene anchors the project, and the whole world spawns imperatively
// from the boot system below. A gritty enclosed room with a wood-block pyramid opens through a doorway
// onto a hall: a plank bridge over a pit, a stack of blocks on it, two wooden chains (one weighted with a
// stone ball). Everything dynamic is grabbable; thrown blocks clack. Materials are procedural: world-space
// grit on the walls, object-space wood grain on the blocks/chains/planks, object-space stone on the weight.
//
// The manifest's `capacity: 512` is load-bearing, not a tidy default: the impact system Mirrors the solver's
// whole persistent contact store every frame (capacity · 3.5 KB — 1.75 MB here; the default 65536 would be
// a 235 MB buffer, unmirrorable).

// the procedural surfaces dirty the per-instance base color through sear's instanced color bindings +
// per-pixel lit(). Walls are world-space grit; props (bricks/rope/bridge) are object-space wood grain and
// the chain weight is object-space stone — object space (localPos × the instance scale) keeps a tumbling
// body's texture fixed to it and uniform in density across the differently-sized props.
const materialLayout = surfaceLayout({
    eids: { type: "storage", element: d.u32 },
    transforms: { type: "storage", element: Xform },
    color: { type: "storage", element: d.u32 },
});

// Dave Hoskins' sin-free integer hash (shadertoy 4djSRW) + trilinear value noise — robust across the whole
// scene where fract(sin(dot())) bands out in f32 once the world coord reaches the ±40 the hall spans.
const hash13 = tgpu.fn(
    [d.vec3f],
    d.f32,
)((p0) => {
    "use gpu";
    let p = d.vec3f(std.fract(std.mul(p0, 0.1031)));
    p = d.vec3f(std.add(p, std.dot(p, std.add(p.zyx, d.vec3f(31.32)))));
    return std.fract((p.x + p.y) * p.z);
});

const valueNoise = tgpu.fn(
    [d.vec3f],
    d.f32,
)((p) => {
    "use gpu";
    const i = std.floor(p);
    const f = std.fract(p);
    const u = std.mul(std.mul(f, f), std.sub(d.vec3f(3), std.mul(f, 2)));
    const c000 = hash13(i);
    const c100 = hash13(std.add(i, d.vec3f(1, 0, 0)));
    const c010 = hash13(std.add(i, d.vec3f(0, 1, 0)));
    const c110 = hash13(std.add(i, d.vec3f(1, 1, 0)));
    const c001 = hash13(std.add(i, d.vec3f(0, 0, 1)));
    const c101 = hash13(std.add(i, d.vec3f(1, 0, 1)));
    const c011 = hash13(std.add(i, d.vec3f(0, 1, 1)));
    const c111 = hash13(std.add(i, d.vec3f(1, 1, 1)));
    return std.mix(
        std.mix(std.mix(c000, c100, u.x), std.mix(c010, c110, u.x), u.y),
        std.mix(std.mix(c001, c101, u.x), std.mix(c011, c111, u.x), u.y),
        u.z,
    );
});

const fbm = tgpu.fn(
    [d.vec3f],
    d.f32,
)((p) => {
    "use gpu";
    return (
        valueNoise(std.mul(p, 3.5)) * 0.55 +
        valueNoise(std.mul(p, 9)) * 0.32 +
        valueNoise(std.mul(p, 20)) * 0.13
    );
});

const gritFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const big = fbm(std.mul(ctx.world, 0.6));
    const mid = fbm(std.mul(ctx.world, 2.3));
    const fine = valueNoise(std.mul(ctx.world, 24));
    const tooth = valueNoise(std.mul(ctx.world, 48));
    const body = big * 0.34 + mid * 0.32 + fine * 0.19 + tooth * 0.15;
    const shade = 0.58 + std.clamp((body - 0.5) * 1.9 + 0.5, 0, 1) * 0.46;
    const mask = std.smoothstep(0.58, 0.78, fbm(std.mul(ctx.world, 0.45)));
    const line = 1 - std.smoothstep(0, 0.02, std.abs(valueNoise(std.mul(ctx.world, 2)) - 0.5));
    const crack = line * mask;
    const tint = std.mix(
        d.vec3f(0.95, 0.97, 1.03),
        d.vec3f(1.05, 1, 0.93),
        fbm(std.mul(ctx.world, 0.4)),
    );
    const base = unpackLdrColor(materialLayout.$.color[ctx.eid]).xyz;
    const albedo = std.mul(std.mul(base, shade * (1 - crack * 0.3)), tint);
    return d.vec4f(lit(albedo, ctx.worldNormal), 1);
});

const woodGrain = tgpu.fn(
    [d.vec3f, d.vec3f],
    d.f32,
)((m, size) => {
    "use gpu";
    let axial = d.f32(m.x);
    let rad = d.vec2f(m.yz);
    if (size.y >= size.x && size.y >= size.z) {
        axial = m.y;
        rad = d.vec2f(m.zx);
    } else if (size.z >= size.x && size.z >= size.y) {
        axial = m.z;
        rad = d.vec2f(m.xy);
    }
    const warp = valueNoise(std.mul(m, 5)) - 0.5;
    const r = std.length(std.sub(rad, d.vec2f(0.4, 2.6))) + warp * 0.22;
    const rings = std.abs(std.fract(r * 22) - 0.5) * 2;
    const fiber = valueNoise(d.vec3f(axial * 3, rad.x * 50, rad.y * 50)) - 0.5;
    return std.clamp(rings * 0.5 + 0.25 + fiber * 0.5, 0, 1);
});

const woodFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const size = materialLayout.$.transforms[ctx.eid].scale;
    const fe = d.f32(ctx.eid);
    const jitter = std.sub(
        d.vec3f(hash13(d.vec3f(fe, 2, 5)), hash13(d.vec3f(fe, 7, 11)), hash13(d.vec3f(fe, 13, 17))),
        d.vec3f(0.5),
    );
    const g = woodGrain(std.add(std.mul(ctx.localPos, size), std.mul(jitter, 4)), size);
    const base = unpackLdrColor(materialLayout.$.color[ctx.eid]).xyz;
    return d.vec4f(lit(std.mul(base, 0.84 + g * 0.2), ctx.worldNormal), 1);
});

const stoneFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const m = std.mul(ctx.localPos, materialLayout.$.transforms[ctx.eid].scale);
    const n = std.clamp((fbm(m) - 0.5) * 2 + 0.5, 0, 1);
    const vein = std.smoothstep(0.42, 0.5, std.abs(valueNoise(std.mul(m, 6)) - 0.5));
    const mottle = (0.74 + n * 0.34) * (1 - vein * 0.35);
    const base = unpackLdrColor(materialLayout.$.color[ctx.eid]).xyz;
    return d.vec4f(lit(std.mul(base, mottle), ctx.worldNormal), 1);
});

function registerSurfaces(state: State): void {
    // GRIT — rough world-space stone for the walls/floor: weathering at three scales (big stains, medium
    // blotches, a fine surface tooth) broken by thin cracks gated to a few weathered patches, plus a subtle
    // warm/cool tint drift, so it reads as aged stone rather than uniform noise. World-space → continuous.
    registerSurface(state, {
        name: "grit",
        layout: materialLayout,
        fs: gritFs,
    });

    // WOOD — subtle, low-contrast grain that stays close to the muted base so it sits in the dim palette.
    // Fibrous streaks along the object's longest axis (the grain) carry the look; a FAR pith makes the
    // growth rings read as near-parallel cathedral lines (not bullseye swirls). A per-eid jitter offsets
    // each block's sample so no two are carbon copies, and grain is a SCALAR darkening (hue unchanged) — no
    // warm split that would pull the wood orange under the lamps.
    registerSurface(state, {
        name: "wood",
        layout: materialLayout,
        fragmentInputs: { localPos: true },
        fs: woodFs,
    });

    // STONE — object-space fbm speckle with a faint pitted vein, color-tuned per instance. Object space
    // (localPos × the instance scale) keeps the grain fixed to the swinging weight.
    registerSurface(state, {
        name: "stone",
        layout: materialLayout,
        fragmentInputs: { localPos: true },
        fs: stoneFs,
    });
}

// ── module refs born in build, cleared on dispose (the module-refs-cleared-on-dispose pattern) ──

let playerEid = -1;
let bodyMirror: Mirror | null = null;
let contactMirror: Mirror | null = null;
let theGun: Gun | null = null;
let booted = false;

// the world spawns from a boot system, not `warm`: build() reads `Avbd.step` (joints, the body Mirror),
// and plugin warms run concurrently (Promise.all in build()), so AvbdPlugin.warm may not have created the
// step yet. A one-shot system gated on `Avbd.step` runs after every plugin has warmed. `setup` re-arms
// per State build.
const BootSystem: System = {
    name: "sandbox-boot",
    group: "simulation",
    setup() {
        booted = false;
    },
    update(state: State) {
        if (booted || !Avbd.step) return;
        booted = true;
        build(state);
        // the crosshair + prompts are gameplay chrome, mounted alongside the scene. Plugin-owned DOM
        // mounts into the engine's sandboxed overlay (`mountOverlay`), the same canvas-bounded surface
        // `config.ui` hands an app. Teardown is State-owned: passing `state` auto-removes the overlay,
        // and the hud's own cleanup registers beside it — both unwind at `state.dispose()`, no plugin
        // `dispose` hook for the UI.
        const overlay = mountOverlay(document.querySelector("canvas"), state);
        state.onDispose(hud(overlay));
        // Pointer Lock has no touch equivalent (this unit ships sandbox desktop-only, `ui.ts`'s
        // header) — a touch-only visitor gets a reason instead of a silently unplayable gun.
        if (isTouchOnly()) state.onDispose(touchNotice(overlay));
    },
};

const GunSystem: System = {
    name: "gun",
    group: "simulation",
    update(state: State) {
        if (!theGun || playerEid < 0) return;
        setCrosshair(theGun.update(state, Player.camera.get(playerEid)));
    },
};

// the volumetric atmosphere — a thin warm haze the whole scene fades into, with the `Volumetric` ceiling
// lamps glowing through it (the hall recedes into haze, each lamp blooms a soft halo, props cut shadow
// shafts). A near-uniform density (slight ground bias) keeps medium up at the ceiling lamps so the glow
// reads. Dialed in against this exact lighting — its fog params are these values.
function addFog(state: State): void {
    const fog = state.create();
    state.add(fog, Fog);
    Fog.density.set(fog, 0.03);
    Fog.color.set(fog, 0x2a2016);
    Fog.heightBase.set(fog, 0);
    Fog.heightFalloff.set(fog, 0.05);
    Fog.absorption.set(fog, 0.1);
    Fog.scattering.set(fog, 4);
    Fog.anisotropy.set(fog, 0.5);
    Fog.scatterIntensity.set(fog, 0.4);
    Fog.steps.set(fog, 32);
    Fog.jitter.set(fog, 1);
}

// G toggles the fog as pure component data: the engine FogSystem no-ops while the `Fog` singleton is
// absent (zero GPU cost), so presence is the on/off switch — destroy it to clear, re-author to restore.
const FogToggleSystem: System = {
    name: "fog-toggle",
    group: "simulation",
    update(state: State) {
        if (!Inputs.isKeyPressed("KeyG")) return;
        const fog = state.only([Fog]);
        if (fog < 0) addFog(state);
        else state.destroy(fog);
    },
};

const SandboxPlugin: Plugin = {
    name: "Sandbox",
    // registerSurfaces registers surfaces; RenderPlugin.initialize clears the render registries in its own
    // initialize, so the dependency orders this plugin's registration after the wipe
    dependencies: [RenderPlugin],
    components: { Brick },
    systems: [BootSystem, GunSystem, ImpactSystem, FogToggleSystem],
    initialize(state) {
        registerSurfaces(state);
        registerInstruments();
    },
    dispose() {
        bodyMirror?.dispose();
        contactMirror?.dispose();
        bodyMirror = null;
        contactMirror = null;
        theGun = null;
        playerEid = -1;
        armImpacts(null);
    },
};

// the manifest references this module by path (`"Sandbox": "./src/sandbox"`) and imports its default
export default SandboxPlugin;

// ── the world ──

// grit palette, lightened ~3× (linear) off the legacy near-black originals (0x161514 / 0x2b2a29 /
// 0x1f1e1d / 0x0e0d0c). On near-black albedo the lamps read only as a harsh hotspot; with surfaces that
// respond to light, a moderate lamp lights the room evenly with a soft falloff.
const FLOOR = hex(0x2b2928);
const CEIL = hex(0x4c4b49);
const WALL = hex(0x393836);
const PIT = hex(0x1e1d1b);

function world(state: State): void {
    // the room: 9×9 footprint, 5 high, a doorway in the front (−z) wall
    box(state, [0, -0.1, 0], [4.5, 0.1, 4.5], 0, FLOOR, "grit");
    box(state, [0, 5.0, 0], [4.5, 0.1, 4.5], 0, CEIL, "grit");
    box(state, [0, 2.5, 4.5], [4.5, 2.5, 0.25], 0, WALL, "grit");
    box(state, [-4.5, 2.5, 0], [0.25, 2.5, 4.5], 0, WALL, "grit");
    box(state, [4.5, 2.5, 0], [0.25, 2.5, 4.5], 0, WALL, "grit");
    box(state, [-2.7, 2.5, -4.5], [1.8, 2.5, 0.25], 0, WALL, "grit");
    box(state, [2.7, 2.5, -4.5], [1.8, 2.5, 0.25], 0, WALL, "grit");
    box(state, [0, 3.95, -4.5], [0.9, 1.05, 0.25], 0, WALL, "grit");

    // the hall beyond the doorway: walls, ceiling, two platforms with a pit between, a step out
    box(state, [-4.5, 1.0, -14.5], [0.25, 4, 10], 0, WALL, "grit");
    box(state, [4.5, 1.0, -14.5], [0.25, 4, 10], 0, WALL, "grit");
    box(state, [0, 1.0, -24.5], [4.5, 4, 0.25], 0, WALL, "grit");
    box(state, [0, 5.0, -14.5], [4.5, 0.1, 10], 0, CEIL, "grit");
    box(state, [0, -1.5, -7.75], [4.5, 1.5, 3.25], 0, FLOOR, "grit");
    box(state, [0, -1.5, -21.75], [4.5, 1.5, 2.75], 0, FLOOR, "grit");
    box(state, [0, -3.1, -15], [4.5, 0.1, 4], 0, PIT, "grit");
    box(state, [2.5, -2.25, -11.5], [1.5, 0.75, 0.5], 0, FLOOR, "grit");

    // lighting: ambient fill + the two ceiling lamps as shadow-casting point lights — the legacy
    // sandbox setup. The room is lit from its own lamps; point shadows are local to each lamp, so the
    // enclosed interior shades from within (a sun shadow map would black out the whole room).
    const ambient = state.create();
    state.add(ambient, AmbientLight);
    AmbientLight.intensity.set(ambient, 0.8);
    lamp(state, [0, 4.9, 0]);
    lamp(state, [0, 4.9, -14.5]);
    // the volumetric atmosphere is opt-in, default off — toggle with G (see FogToggleSystem)
}

// spawn the world + props + player — runs once from BootSystem after the physics step exists. Idempotent
// per State: a rebuild re-runs it, re-creating the derived bodies (they live in State, not the serialized scene).
function build(state: State): void {
    const step = Avbd.step;
    if (!step) throw new Error("[sandbox] AvbdPlugin not warmed — no step");
    const backend = Physics.backend;
    if (!backend) throw new Error("[sandbox] no physics backend installed");

    world(state);
    pyramid(state, 0, 0, -4.0, 10);

    const joints: JointDef[] = [];
    rope(state, -2.5, 4.9, -15, 0, joints);
    rope(state, 2.5, 4.9, -15, 0.8, joints);
    bridge(state, 0, -0.1, -15, joints);
    brickStack(state, 0, -15, 5, 0);
    step.setJoints(joints);

    // the player: a kinematic capsule the character controller drives; the camera follows first-person
    // and carries the spatial-audio listener
    const body = state.create();
    state.add(body, Body);
    Body.shape.set(body, ShapeKind.Capsule);
    Body.pos.set(body, 0, 0.9, 3.0, 0);
    Body.halfExtents.set(body, 0, 0.5, 0, 0.4);
    Body.mass.set(body, 0);
    Body.friction.set(body, 0.8);
    state.add(body, Character);
    Character.jumpSpeed.set(body, 15.8);
    Character.gravity.set(body, -50);
    state.add(body, Player);
    playerEid = body;

    const cam = state.create();
    state.add(cam, Transform);
    state.add(cam, Camera);
    state.add(cam, Sear);
    state.add(cam, Depth); // sear's depth lane — the gun outline's occlude gate samples it
    state.add(cam, Listener);
    Camera.mode.set(cam, CameraMode.Perspective);
    Camera.fov.set(cam, 75);
    Camera.near.set(cam, 0.05);
    Camera.far.set(cam, 100);
    // PSX-modern: render at a fixed 360 lines (width follows the canvas aspect) and point-sample up, no
    // MSAA — crisp low-res. posterize to 10 OkLab-L bands; the dither is one band wide (1/10) so Bayer4's
    // ±0.5 range spans a band boundary — the amplitude that fully breaks the banding.
    Camera.antialias.set(cam, 0);
    state.add(cam, Resolution);
    Resolution.height.set(cam, 360);
    state.add(cam, Glaze);
    Glaze.vignette.set(cam, 0.15);
    Glaze.posterize.set(cam, 10);
    Glaze.dither.set(cam, 0.1);
    // PSX-modern color grade: a scene-referred CDL warms the palette (slope) and crushes the blacks
    // (offset + power) before the tonemap, then a slight desaturation after. Starter values for the
    // dim-warm reference look — tuned by eye in the re-dress.
    Glaze.slope.set(cam, 1.06, 1.0, 0.94, 0);
    Glaze.offset.set(cam, -0.01, -0.01, -0.01, 0);
    Glaze.power.set(cam, 1.12, 1.12, 1.12, 0);
    Glaze.saturation.set(cam, 0.92);
    Player.camera.set(body, cam);

    bodyMirror = mirror(step.bodies);
    contactMirror = mirror(step.pairContacts);
    theGun = gun(step, backend, joints, (eid) => eid === playerEid);
    armImpacts({ step, contacts: contactMirror, bodies: bodyMirror });
}
