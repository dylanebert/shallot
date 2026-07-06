// Sky — opt-in procedural sky. A plugin behind sear's backdrop seam: it registers a `Backgrounds` recipe
// (the bryce3d view-ray → HDR color fragment, in `./shader`) and publishes one uniform buffer the recipe
// reads. The engine names no sky concept — this plugin owns all of it. It *reads* the sun from the
// `Lighting` singleton and writes nothing; a day-night cycle that writes the sun is a separate, deferred
// plugin, so sky and lights never depend on each other. One `Sky` singleton holds the look; a camera opts
// in with sear's `Backdrop` component (`backdrop="name: sky"`). Not in `DEFAULT_PLUGINS`.
import type { Plugin, System } from "../../engine";
import { Compute, f32, formatHex, sparse } from "../../engine";
import { RenderPlugin } from "../../standard/render";
import { BeginFrameSystem } from "../../standard/render/core";
import { SearPlugin } from "../../standard/sear";
import { Backgrounds, ColorSystem } from "../../standard/sear/core";
import { packSky } from "./pack";
import { SKY_BYTES, SKY_FLOATS, SKY_WGSL } from "./shader";

/**
 * the scene's procedural sky — one per scene (a singleton). A camera shows it by selecting the registered
 * `sky` backdrop (`backdrop="name: sky"`). The look is a layered recipe: an elevation gradient from
 * `horizon` up to `zenith`, a sun glow + disk (positioned by the scene's directional light, tinted
 * `sun-color`), FBM `cloud`s, hash-grid `star`s, and a `haze` band fading the horizon. The sun's direction
 * follows the directional light; this component sets only its appearance.
 *
 * @example
 * ```
 * <a sky="zenith: 0x89b6e9; horizon: 0xc4cdda; sun-glow: 0.5; cloud-coverage: 0.5" />
 * <a camera sear backdrop="name: sky" transform />
 * ```
 */
export const Sky = {
    /** hex sRGB color overhead, at the zenith (e.g. 0x89b6e9) */
    zenith: sparse(f32),
    /** hex sRGB color at the horizon, blended up toward `zenith` */
    horizon: sparse(f32),
    /** bright band strength right at the horizon line [0,1] (0 = none) */
    band: sparse(f32),
    /** hex sRGB tint of the sun glow + disk (the sun's *position* follows the directional light) */
    sunColor: sparse(f32),
    /** sun disk size [0,1] — larger paints a bigger disk */
    sunSize: sparse(f32),
    /** sun glow strength around the disk [0,1] (0 = no glow) */
    sunGlow: sparse(f32),
    /** hex sRGB cloud color */
    cloudColor: sparse(f32),
    /** cloud coverage [0,1] — how much of the sky the clouds fill (0 = clear) */
    cloudCoverage: sparse(f32),
    /** cloud opacity / thickness [0,1] */
    cloudDensity: sparse(f32),
    /** cloud layer height — scales the projected cloud size (larger = higher, smaller clouds) */
    cloudHeight: sparse(f32),
    /** star brightness [0,1] (0 = no stars) */
    starIntensity: sparse(f32),
    /** star density [0,1] — more stars in the grid */
    starAmount: sparse(f32),
    /** hex sRGB haze color the horizon fades toward */
    hazeColor: sparse(f32),
    /** horizon haze strength [0,1] (0 = none) */
    hazeDensity: sparse(f32),
};

let _buffer: GPUBuffer | null = null;
const _staging = new Float32Array(SKY_FLOATS);

// writes the `Sky` uniform each frame from the scene's Sky singleton, before sear's color pass reads it for
// the backdrop draw. No-op unless the scene has a Sky singleton.
const SkySystem: System = {
    name: "sky",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [ColorSystem],
    update(state) {
        const device = Compute.device;
        if (!device || !_buffer) return;
        const eid = state.only([Sky]);
        if (eid < 0) return;
        packSky(eid, _staging);
        device.queue.writeBuffer(_buffer, 0, _staging as Float32Array<ArrayBuffer>);
    },
};

/**
 * procedural sky (the bryce3d look). Opt-in: add `SkyPlugin` to the plugin set, give the scene one
 * {@link Sky} singleton, and select the `sky` backdrop on the rendering camera (`backdrop="name: sky"`).
 * The sky reads the scene's directional light for the sun's position and writes nothing.
 */
export const SkyPlugin: Plugin = {
    name: "Sky",
    components: { Sky },
    traits: {
        Sky: {
            singleton: true,
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
            format: {
                zenith: formatHex,
                horizon: formatHex,
                sunColor: formatHex,
                cloudColor: formatHex,
                hazeColor: formatHex,
            },
        },
    },
    systems: [SkySystem],
    // SearPlugin so this initialize runs after SearPlugin clears the Backgrounds registry; RenderPlugin for
    // the Lighting uniform the fragment reads
    dependencies: [RenderPlugin, SearPlugin],

    initialize() {
        Backgrounds.register({
            name: "sky",
            bindings: { sky: { type: "uniform", struct: "Sky" } },
            preamble: SKY_WGSL,
            fs: "col = sampleSky(dir);",
        });
    },

    warm() {
        const { device } = Compute;
        if (!device) return;
        _buffer?.destroy();
        _buffer = device.createBuffer({
            label: "sky-config",
            size: SKY_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // the background bind group resolves the `sky` binding from Compute.buffers by name; republish every
        // warm — the map is wiped on each build()
        Compute.buffers.set("sky", _buffer);
    },

    dispose() {
        _buffer?.destroy();
        _buffer = null;
    },
};
