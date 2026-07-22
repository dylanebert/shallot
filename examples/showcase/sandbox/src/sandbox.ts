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
import { type Binding, Surfaces } from "@dylanebert/shallot/render/core";
import { armImpacts, ImpactSystem, registerInstruments } from "./audio";
import { type Gun, gun } from "./gun";
import { Brick, box, brickStack, bridge, hex, lamp, pyramid, rope } from "./spawn";
import { hud, setCrosshair } from "./ui";

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
const INSTANCED: Record<string, Binding> = {
    eids: { type: "storage", element: "u32" },
    transforms: { type: "storage", element: "Xform" },
    color: { type: "storage", element: "u32" },
};

// Dave Hoskins' sin-free integer hash (shadertoy 4djSRW) + trilinear value noise — robust across the whole
// scene where fract(sin(dot())) bands out in f32 once the world coord reaches the ±40 the hall spans.
const NOISE_WGSL = /* wgsl */ `
    fn hash13(p0: vec3<f32>) -> f32 {
        var p = fract(p0 * 0.1031);
        p += dot(p, p.zyx + 31.32);
        return fract((p.x + p.y) * p.z);
    }
    fn valueNoise(p: vec3<f32>) -> f32 {
        let i = floor(p);
        let f = fract(p);
        let u = f * f * (3.0 - 2.0 * f);
        let c000 = hash13(i);
        let c100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
        let c010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
        let c110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
        let c001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
        let c101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
        let c011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
        let c111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));
        return mix(
            mix(mix(c000, c100, u.x), mix(c010, c110, u.x), u.y),
            mix(mix(c001, c101, u.x), mix(c011, c111, u.x), u.y), u.z);
    }
    fn fbm(p: vec3<f32>) -> f32 {
        return valueNoise(p * 3.5) * 0.55 + valueNoise(p * 9.0) * 0.32 + valueNoise(p * 20.0) * 0.13;
    }
`;

function registerSurfaces(): void {
    // GRIT — rough world-space stone for the walls/floor: weathering at three scales (big stains, medium
    // blotches, a fine surface tooth) broken by thin cracks gated to a few weathered patches, plus a subtle
    // warm/cool tint drift, so it reads as aged stone rather than uniform noise. World-space → continuous.
    Surfaces.register({
        name: "grit",
        bindings: INSTANCED,
        preamble: NOISE_WGSL,
        fs: /* wgsl */ `
            let big = fbm(world * 0.6);
            let mid = fbm(world * 2.3);
            let fine = valueNoise(world * 24.0);
            let tooth = valueNoise(world * 48.0);
            let body = big * 0.34 + mid * 0.32 + fine * 0.19 + tooth * 0.15;
            let shade = 0.58 + clamp((body - 0.5) * 1.9 + 0.5, 0.0, 1.0) * 0.46;
            // a noise iso-contour tiles the whole plane with closed loops (reads as cracked mud), so a
            // low-frequency mask concentrates the thin dark fractures into a few weathered patches
            let mask = smoothstep(0.58, 0.78, fbm(world * 0.45));
            let line = 1.0 - smoothstep(0.0, 0.02, abs(valueNoise(world * 2.0) - 0.5));
            let crack = line * mask;
            let tint = mix(vec3<f32>(0.95, 0.97, 1.03), vec3<f32>(1.05, 1.0, 0.93), fbm(world * 0.4));
            let albedo = unpackLdrColor(color[eid]).rgb * shade * (1.0 - crack * 0.3) * tint;
            col = vec4<f32>(lit(albedo, worldNormal), 1.0);
        `,
    });

    // WOOD — subtle, low-contrast grain that stays close to the muted base so it sits in the dim palette.
    // Fibrous streaks along the object's longest axis (the grain) carry the look; a FAR pith makes the
    // growth rings read as near-parallel cathedral lines (not bullseye swirls). A per-eid jitter offsets
    // each block's sample so no two are carbon copies, and grain is a SCALAR darkening (hue unchanged) — no
    // warm split that would pull the wood orange under the lamps.
    Surfaces.register({
        name: "wood",
        bindings: INSTANCED,
        preamble:
            NOISE_WGSL +
            /* wgsl */ `
            fn woodGrain(m: vec3<f32>, size: vec3<f32>) -> f32 {
                var axial = m.x; var rad = m.yz;
                if (size.y >= size.x && size.y >= size.z) { axial = m.y; rad = m.zx; }
                else if (size.z >= size.x && size.z >= size.y) { axial = m.z; rad = m.xy; }
                let warp = valueNoise(m * 5.0) - 0.5;
                let r = length(rad - vec2<f32>(0.4, 2.6)) + warp * 0.22;
                let rings = abs(fract(r * 22.0) - 0.5) * 2.0;
                let fiber = valueNoise(vec3<f32>(axial * 3.0, rad.x * 50.0, rad.y * 50.0)) - 0.5;
                return clamp(rings * 0.5 + 0.25 + fiber * 0.5, 0.0, 1.0);
            }
        `,
        fs: /* wgsl */ `
            let size = transforms[eid].scale;
            let fe = f32(eid);
            let j = vec3<f32>(hash13(vec3<f32>(fe, 2.0, 5.0)),
                              hash13(vec3<f32>(fe, 7.0, 11.0)),
                              hash13(vec3<f32>(fe, 13.0, 17.0))) - 0.5;
            let g = woodGrain(localPos * size + j * 4.0, size);
            let albedo = unpackLdrColor(color[eid]).rgb * (0.84 + g * 0.2);
            col = vec4<f32>(lit(albedo, worldNormal), 1.0);
        `,
    });

    // STONE — object-space fbm speckle with a faint pitted vein, color-tuned per instance. Object space
    // (localPos × the instance scale) keeps the grain fixed to the swinging weight.
    Surfaces.register({
        name: "stone",
        bindings: INSTANCED,
        preamble: NOISE_WGSL,
        fs: /* wgsl */ `
            let m = localPos * transforms[eid].scale;
            let n = clamp((fbm(m) - 0.5) * 2.0 + 0.5, 0.0, 1.0);
            let vein = smoothstep(0.42, 0.5, abs(valueNoise(m * 6.0) - 0.5));
            let mottle = (0.74 + n * 0.34) * (1.0 - vein * 0.35);
            col = vec4<f32>(lit(unpackLdrColor(color[eid]).rgb * mottle, worldNormal), 1.0);
        `,
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
// step yet. A one-shot system gated on `Avbd.step` runs after every plugin has warmed — `mode: always` so
// the room renders in edit mode too, not only play. `setup` re-arms per State build.
const BootSystem: System = {
    name: "sandbox-boot",
    group: "simulation",
    annotations: { mode: "always" },
    setup() {
        booted = false;
    },
    update(state: State) {
        if (booted || !Avbd.step) return;
        booted = true;
        build(state);
        // the crosshair + prompts are gameplay chrome — skip them in edit mode (the gun
        // doesn't run there), mount in play + standalone. Plugin-owned DOM mounts into the engine's
        // sandboxed overlay (`mountOverlay`), the same canvas-bounded surface `config.ui` hands an app.
        // Teardown is State-owned: passing `state` auto-removes the overlay, and the hud's own cleanup
        // registers beside it — both unwind at `state.dispose()`, no plugin `dispose` hook for the UI.
        if (state.mode !== "edit") {
            const overlay = mountOverlay(document.querySelector("canvas"), state);
            state.onDispose(hud(overlay));
        }
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
    initialize() {
        registerSurfaces();
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
// per State: a rebuild re-runs it, re-creating the derived bodies (they live in State, not the Document).
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
