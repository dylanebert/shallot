import type { World } from "../../engine";
import { component } from "../../engine";
// Sky reads directional lights without changing them; atmosphere does not own the light state.

import { BeginFrameSystem, MainPassSystem, RenderingPlugin, SunDisk } from "../../core/rendering";
import type { Plugin, System } from "../../engine";
import { f32 } from "../../engine";
import {
    type Background,
    type BackgroundHandle,
    registerBackground,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { packSky } from "./pack";
import { SKY_BYTES, SKY_FLOATS, SkyGpu, skyBackground } from "./shader";

/**
 * the scene's procedural sky, one per scene (a singleton). A camera shows it by selecting the registered
 * handle from `skyBackgroundHandle(world)` with `CameraBackground.background`. The look is a layered recipe: an elevation gradient from
 * `horizon` up to `zenith`, a sun glow + disk (positioned by the scene's directional light, tinted
 * per-light `SunDisk`s), FBM `cloud`s, hash-grid `star`s, and a `haze` band fading the horizon. Each disk
 * follows its directional light and takes that light's color.
 */
export const Sky = component(
    "Sky",
    {
        /** Scale of the complete sky, including SunDisk disks and glow, in cd/m².
         * Multiplied by each view's exposure. The default `1.2 * 2 ** 9.7` is a tuned look
         * value that gives unit gain at the default Exposure, not a measured sky luminance.
         */
        brightness: f32,
        /** hex sRGB color overhead, at the zenith (e.g. 0x89b6e9) */
        zenith: f32,
        /** hex sRGB color at the horizon, blended up toward `zenith` */
        horizon: f32,
        /** bright band strength right at the horizon line [0,1] (0 = none) */
        band: f32,
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
    },
    {
        defaults: () => ({
            brightness: 1.2 * 2 ** 9.7,
            zenith: 0x89b6e9,
            horizon: 0xc4cdda,
            band: 0,
            cloudColor: 0xffffff,
            cloudCoverage: 0.5,
            cloudDensity: 0.7,
            cloudHeight: 4,
            starIntensity: 0,
            starAmount: 0.5,
            hazeColor: 0xbcc5d4,
            hazeDensity: 0.005,
        }),
    },
);

interface SkyState {
    buffer: GPUBuffer | null;
    bindings: ReturnType<NonNullable<Background["bindings"]>>;
    background: BackgroundHandle | null;
    staging: Float32Array;
}

const skyStateKey = { create: () => createSkyState() };
const createSkyState = (): SkyState => ({
    buffer: null,
    bindings: {},
    background: null,
    staging: new Float32Array(SKY_FLOATS),
});
const skyState = (world: World) => world.resource(skyStateKey);

/** The World-local handle for CameraBackground.background. Refuses before SkyPlugin initializes. */
export function skyBackgroundHandle(world: World): BackgroundHandle {
    const handle = skyState(world).background;
    if (handle === null)
        throw new Error("skyBackgroundHandle requires SkyPlugin to initialize first");
    return handle;
}

const SkySystem: System = {
    name: "sky",
    group: "draw",
    after: [BeginFrameSystem],
    before: [MainPassSystem],
    update(world) {
        const device = world.gpu.device;
        const sky = skyState(world);
        if (!sky.buffer) return;
        const eid = world.only([Sky]);
        if (eid < 0) return;
        packSky(world, eid, sky.staging);
        device.queue.writeBuffer(sky.buffer, 0, sky.staging as Float32Array<ArrayBuffer>);
    },
};

/**
 * procedural sky (the bryce3d look). Opt-in: add `SkyPlugin` to the plugin set, give the scene one
 * {@link Sky} singleton, add one or more {@link SunDisk}s to directional lights, and select
 * `skyBackgroundHandle(world)` on the rendering camera with `CameraBackground.background`. The sky reads each opted-in light and writes nothing.
 */
export const SkyPlugin: Plugin = {
    gpu: {},
    name: "Sky",
    components: [Sky],

    systems: [SkySystem],
    dependencies: [RenderingPlugin, StandardRenderingPlugin],

    initialize(world) {
        skyState(world).background = registerBackground(world, {
            ...skyBackground,
            bindings: (world) => skyState(world).bindings,
        });
    },

    warm(world: World) {
        const { device } = world.gpu;
        const sky = skyState(world);
        sky.buffer?.destroy();
        sky.buffer = device.createBuffer({
            label: "sky-config",
            size: SKY_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        sky.bindings.sky = world.gpu.root
            .createBuffer(SkyGpu, sky.buffer)
            .$usage("uniform")
            .$name("sky-config");
    },

    dispose(world) {
        const sky = world.resource(skyStateKey);
        sky.buffer?.destroy();
        sky.buffer = null;
    },
};
