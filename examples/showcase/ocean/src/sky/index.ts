import {
    Compute,
    f32,
    formatHex,
    type Plugin,
    RenderPlugin,
    SearPlugin,
    type System,
    sparse,
    unpackColor,
} from "@dylanebert/shallot";
import { BeginFrameSystem } from "@dylanebert/shallot/render/core";
import { ColorSystem, registerBackground } from "@dylanebert/shallot/sear/core";
import { DUSK_SKY_BYTES, DUSK_SKY_FLOATS, DuskSkyGpu, skyBackground } from "./shader";

/** Reference-shaped defaults for the ocean showcase's dusk sky. */
export const DUSK_SKY_DEFAULTS = {
    zenith: 0x31577e,
    horizon: 0xd5a6a0,
    haze: 0xe8b5a4,
    hazeStrength: 0.16,
    cloud: 0x715f78,
    cloudStrength: 0.18,
    sun: 0xffd5a0,
    sunStrength: 0.42,
    exposure: 1.05,
} as const;

/** The ocean showcase's one demo-local sky singleton. */
export const Sky = {
    zenith: sparse(f32),
    horizon: sparse(f32),
    haze: sparse(f32),
    hazeStrength: sparse(f32),
    cloud: sparse(f32),
    cloudStrength: sparse(f32),
    sun: sparse(f32),
    sunStrength: sparse(f32),
    exposure: sparse(f32),
};

let buffer: GPUBuffer | null = null;
const staging = new Float32Array(DUSK_SKY_FLOATS);

function color(out: Float32Array, offset: number, value: number, strength = 0): void {
    const decoded = unpackColor(value);
    out[offset] = decoded.r;
    out[offset + 1] = decoded.g;
    out[offset + 2] = decoded.b;
    out[offset + 3] = strength;
}

/** Packs the scene singleton into the GPU contract. */
export function packSky(eid: number, out = staging): Float32Array {
    out.fill(0);
    color(out, 0, Sky.zenith.get(eid));
    color(out, 4, Sky.horizon.get(eid));
    color(out, 8, Sky.haze.get(eid), Sky.hazeStrength.get(eid));
    color(out, 12, Sky.cloud.get(eid), Sky.cloudStrength.get(eid));
    color(out, 16, Sky.sun.get(eid), Sky.sunStrength.get(eid));
    out[20] = Sky.exposure.get(eid);
    return out;
}

const SkySystem: System = {
    name: "ocean-dusk-sky",
    group: "draw",
    after: [BeginFrameSystem],
    before: [ColorSystem],
    update(state) {
        if (!buffer) return;
        const eid = state.only([Sky]);
        if (eid < 0) return;
        Compute.device?.queue.writeBuffer(buffer, 0, packSky(eid));
    },
};

/** Separate demo-local dusk sky plugin. */
export const SkyPlugin: Plugin = {
    name: "Sky",
    components: { Sky },
    traits: {
        Sky: {
            singleton: true,
            defaults: () => ({ ...DUSK_SKY_DEFAULTS }),
            format: {
                zenith: formatHex,
                horizon: formatHex,
                haze: formatHex,
                cloud: formatHex,
                sun: formatHex,
            },
        },
    },
    systems: [SkySystem],
    dependencies: [RenderPlugin, SearPlugin],
    initialize(state) {
        registerBackground(state, { ...skyBackground });
    },
    warm() {
        buffer?.destroy();
        buffer = Compute.device.createBuffer({
            label: "ocean-dusk-sky",
            size: DUSK_SKY_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        Compute.buffers.set("duskSky", buffer);
        Compute.typed.set(
            "duskSky",
            Compute.root.createBuffer(DuskSkyGpu, buffer).$usage("uniform").$name("ocean-dusk-sky"),
        );
    },
    dispose() {
        buffer?.destroy();
        buffer = null;
    },
};

export {
    DuskSkyGpu,
    SOLAR_ANGULAR_RADIUS,
    SOLAR_LIMB_EXPONENT,
    sampleCloud,
    sampleElevation,
    sampleHaze,
    sampleSky,
    sampleSun,
    solarDiskProfile,
} from "./shader";
