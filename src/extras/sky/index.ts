import type { World } from "../../engine";
import { registration } from "../../engine";
// Sky — opt-in procedural sky. A plugin behind sear's backdrop seam: it registers a `Backgrounds` recipe
// (the bryce3d view-ray → HDR color fragment, in `./shader`) and publishes one uniform buffer the recipe
// reads. The engine names no sky concept — this plugin owns all of it. It *reads* the sun from the
// `Lighting` singleton and writes nothing; a day-night cycle that writes the sun is a separate, deferred
// plugin, so sky and lights never depend on each other. One `Sky` singleton holds the look; a camera opts
// in with sear's `CameraBackground` component. Not in `DEFAULT_PLUGINS`.

import { BeginFrameSystem, RenderPlugin, registerBackground } from "../../core/rendering";
import type { Plugin, System } from "../../engine";
import { f32 } from "../../engine";

import { RenderMeshColorSystem, SearPlugin } from "../../standard/rendering";
import { packSky } from "./pack";
import { SKY_BYTES, SKY_FLOATS, SkyGpu, skyBackground } from "./shader";

/**
 * the scene's procedural sky, one per scene (a singleton). A camera shows it by selecting the registered
 * `sky` background with `CameraBackground`. The look is a layered recipe: an elevation gradient from
 * `horizon` up to `zenith`, a sun glow + disk (positioned by the scene's directional light, tinted
 * `sunColor`), FBM `cloud`s, hash-grid `star`s, and a `haze` band fading the horizon. The sun's direction
 * follows the directional light; this component sets only its appearance.
 *
 * @example
 * ```
 * world.add(world.create(), Sky, { zenith: 0x89b6e9, horizon: 0xc4cdda, sunGlow: 0.5, cloudCoverage: 0.5 });
 * const camera = world.create();
 * world.add(camera, Camera);
 * world.add(camera, StandardRenderer);
 * world.add(camera, CameraBackground, { name: world.resource(Backgrounds).id("sky") ?? 0 });
 * world.add(camera, Transform);
 * ```
 */
export const Sky = {
    /** hex sRGB color overhead, at the zenith (e.g. 0x89b6e9) */
    zenith: f32,
    /** hex sRGB color at the horizon, blended up toward `zenith` */
    horizon: f32,
    /** bright band strength right at the horizon line [0,1] (0 = none) */
    band: f32,
    /** hex sRGB tint of the sun glow + disk (the sun's *position* follows the directional light) */
    sunColor: f32,
    /** sun disk size [0,1]: larger paints a bigger disk */
    sunSize: f32,
    /** sun glow strength around the disk [0,1] (0 = no glow) */
    sunGlow: f32,
    /** hex sRGB cloud color */
    cloudColor: f32,
    /** cloud coverage [0,1]: how much of the sky the clouds fill (0 = clear) */
    cloudCoverage: f32,
    /** cloud opacity / thickness [0,1] */
    cloudDensity: f32,
    /** cloud layer height: scales the projected cloud size (larger = higher, smaller clouds) */
    cloudHeight: f32,
    /** star brightness [0,1] (0 = no stars) */
    starIntensity: f32,
    /** star density [0,1]: more stars in the grid */
    starAmount: f32,
    /** hex sRGB haze color the horizon fades toward */
    hazeColor: f32,
    /** horizon haze strength [0,1] (0 = none) */
    hazeDensity: f32,
};

interface SkyState {
    buffer: GPUBuffer | null;
    staging: Float32Array;
}

const skyStateKey = { create: () => createSkyState() };
const createSkyState = (): SkyState => ({ buffer: null, staging: new Float32Array(SKY_FLOATS) });
const skyState = (world: World) => world.resource(skyStateKey);

// writes the `Sky` uniform each frame from the scene's Sky singleton, before sear's color pass reads it for
// the backdrop draw. No-op unless the scene has a Sky singleton.
const SkySystem: System = {
    name: "sky",
    group: "draw",
    after: [BeginFrameSystem],
    before: [RenderMeshColorSystem],
    update(world) {
        const device = world.gpu.device;
        const sky = skyState(world);
        if (!device || !sky.buffer) return;
        const eid = world.only([Sky]);
        if (eid < 0) return;
        packSky(world, eid, sky.staging);
        device.queue.writeBuffer(sky.buffer, 0, sky.staging as Float32Array<ArrayBuffer>);
    },
};

/**
 * procedural sky (the bryce3d look). Opt-in: add `SkyPlugin` to the plugin set, give the scene one
 * {@link Sky} singleton, and select the `sky` background on the rendering camera with `CameraBackground`.
 * The sky reads the scene's directional light for the sun's position and writes nothing.
 */
export const SkyPlugin: Plugin = {
    name: "Sky",
    components: [
        registration("Sky", Sky, {
            defaults: () => ({
                zenith: 0x89b6e9,
                horizon: 0xc4cdda,
                band: 0,
                sunColor: 0xffffff,
                sunSize: 0.7,
                sunGlow: 0.5,
                cloudColor: 0xffffff,
                cloudCoverage: 0.5,
                cloudDensity: 0.7,
                cloudHeight: 4,
                starIntensity: 0,
                starAmount: 0.5,
                hazeColor: 0xbcc5d4,
                hazeDensity: 0.005,
            }),
        }),
    ],

    systems: [SkySystem],
    // SearPlugin so this initialize runs after SearPlugin clears the Backgrounds registry; RenderPlugin for
    // the Lighting uniform the fragment reads
    dependencies: [RenderPlugin, SearPlugin],

    initialize(world) {
        world.resource(skyStateKey);
        // Each World owns a distinct spec identity. Reusing the module singleton here would let an old
        // World's exact-object disposal guard mistake a later build for its own registration.
        registerBackground(world, { ...skyBackground });
    },

    warm(world: World) {
        const { device } = world.gpu;
        if (!device) return;
        const sky = skyState(world);
        sky.buffer?.destroy();
        sky.buffer = device.createBuffer({
            label: "sky-config",
            size: SKY_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // the background bind group resolves the `sky` binding from state.gpu.buffers by name; republish every
        // warm — the map is wiped on each createApp()
        world.gpu.buffers.set("sky", sky.buffer);
        world.gpu.typed.set(
            "sky",
            world.gpu.root.createBuffer(SkyGpu, sky.buffer).$usage("uniform").$name("sky-config"),
        );
    },

    dispose(world) {
        const sky = world.resource(skyStateKey);
        sky.buffer?.destroy();
        sky.buffer = null;
    },
};
