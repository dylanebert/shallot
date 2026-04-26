import { linearToSrgb, traits, type Plugin, type State, type System } from "../../engine";
import {
    Sky,
    Sun,
    Stars,
    Moon,
    Haze,
    Clouds,
    AmbientLight,
    DirectionalLight,
} from "../../standard/render";

export const Skylab = {
    azimuth: [] as number[],
    elevation: [] as number[],
    skyColor: [] as number[],
    horizonColor: [] as number[],
};
traits(Skylab, { defaults: () => ({ azimuth: 37, elevation: 45, skyColor: 0, horizonColor: 0 }) });

interface ColorRGB {
    r: number;
    g: number;
    b: number;
}

interface GradientStop {
    elevation: number;
    sunColor: ColorRGB;
    sunIntensity: number;
    ambientColor: ColorRGB;
    ambientIntensity: number;
    zenith: ColorRGB;
    horizon: ColorRGB;
    sunDisk: number;
    sunGlow: number;
    starsIntensity: number;
    moonDisk: number;
    moonElevationOffset: number;
    hazeDensity: number;
    hazeColor: ColorRGB;
    cloudsColor: ColorRGB;
}

const STOPS: GradientStop[] = [
    {
        elevation: -90,
        sunColor: { r: 0.25, g: 0.3, b: 0.55 },
        sunIntensity: 0.15,
        ambientColor: { r: 0.3, g: 0.25, b: 0.55 },
        ambientIntensity: 0.5,
        zenith: { r: 0.03, g: 0.02, b: 0.1 },
        horizon: { r: 0.05, g: 0.06, b: 0.12 },
        sunDisk: 0,
        sunGlow: 0,
        starsIntensity: 1.0,
        moonDisk: 1.0,
        moonElevationOffset: 180,
        hazeDensity: 0.001,
        hazeColor: { r: 0.05, g: 0.04, b: 0.12 },
        cloudsColor: { r: 0.1, g: 0.1, b: 0.18 },
    },
    {
        elevation: -18,
        sunColor: { r: 0.6, g: 0.2, b: 0.3 },
        sunIntensity: 0.1,
        ambientColor: { r: 0.25, g: 0.18, b: 0.45 },
        ambientIntensity: 0.5,
        zenith: { r: 0.05, g: 0.02, b: 0.15 },
        horizon: { r: 0.4, g: 0.15, b: 0.25 },
        sunDisk: 0,
        sunGlow: 0,
        starsIntensity: 0.6,
        moonDisk: 0.8,
        moonElevationOffset: 160,
        hazeDensity: 0.003,
        hazeColor: { r: 0.25, g: 0.1, b: 0.2 },
        cloudsColor: { r: 0.3, g: 0.12, b: 0.2 },
    },
    {
        elevation: 0,
        sunColor: { r: 1.0, g: 0.45, b: 0.15 },
        sunIntensity: 0.4,
        ambientColor: { r: 0.35, g: 0.2, b: 0.15 },
        ambientIntensity: 0.5,
        zenith: { r: 0.1, g: 0.08, b: 0.3 },
        horizon: { r: 0.95, g: 0.45, b: 0.15 },
        sunDisk: 0.7,
        sunGlow: 0.5,
        starsIntensity: 0.1,
        moonDisk: 0,
        moonElevationOffset: 140,
        hazeDensity: 0.008,
        hazeColor: { r: 0.85, g: 0.4, b: 0.15 },
        cloudsColor: { r: 0.95, g: 0.5, b: 0.2 },
    },
    {
        elevation: 8,
        sunColor: { r: 1.0, g: 0.7, b: 0.4 },
        sunIntensity: 0.7,
        ambientColor: { r: 0.5, g: 0.4, b: 0.3 },
        ambientIntensity: 0.7,
        zenith: { r: 0.15, g: 0.25, b: 0.55 },
        horizon: { r: 0.85, g: 0.6, b: 0.3 },
        sunDisk: 0.7,
        sunGlow: 0.3,
        starsIntensity: 0,
        moonDisk: 0,
        moonElevationOffset: 120,
        hazeDensity: 0.006,
        hazeColor: { r: 0.75, g: 0.55, b: 0.3 },
        cloudsColor: { r: 0.95, g: 0.75, b: 0.5 },
    },
    {
        elevation: 20,
        sunColor: { r: 1.0, g: 1.0, b: 1.0 },
        sunIntensity: 0.75,
        ambientColor: { r: 0.55, g: 0.52, b: 0.5 },
        ambientIntensity: 1.0,
        zenith: { r: 0.25, g: 0.48, b: 0.82 },
        horizon: { r: 0.52, g: 0.58, b: 0.68 },
        sunDisk: 0.7,
        sunGlow: 0.6,
        starsIntensity: 0,
        moonDisk: 0,
        moonElevationOffset: 90,
        hazeDensity: 0.005,
        hazeColor: { r: 0.48, g: 0.54, b: 0.64 },
        cloudsColor: { r: 1.0, g: 1.0, b: 1.0 },
    },
    {
        elevation: 50,
        sunColor: { r: 1.0, g: 1.0, b: 1.0 },
        sunIntensity: 0.81,
        ambientColor: { r: 0.53, g: 0.536, b: 0.54 },
        ambientIntensity: 1.0,
        zenith: { r: 0.25, g: 0.47, b: 0.815 },
        horizon: { r: 0.55, g: 0.61, b: 0.7 },
        sunDisk: 0.7,
        sunGlow: 0.5,
        starsIntensity: 0,
        moonDisk: 0,
        moonElevationOffset: 60,
        hazeDensity: 0.005,
        hazeColor: { r: 0.5, g: 0.56, b: 0.66 },
        cloudsColor: { r: 1.0, g: 1.0, b: 1.0 },
    },
];

function lerpColor(a: ColorRGB, b: ColorRGB, t: number): ColorRGB {
    return {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
    };
}

function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

function packColor(c: ColorRGB): number {
    const r = Math.round(linearToSrgb(Math.min(1, Math.max(0, c.r))) * 255);
    const g = Math.round(linearToSrgb(Math.min(1, Math.max(0, c.g))) * 255);
    const b = Math.round(linearToSrgb(Math.min(1, Math.max(0, c.b))) * 255);
    return (r << 16) | (g << 8) | b;
}

export interface SkylabOutput {
    sunColor: number;
    sunIntensity: number;
    ambientColor: number;
    ambientIntensity: number;
    zenith: number;
    horizon: number;
    sunDisk: number;
    sunGlow: number;
    starsIntensity: number;
    moonDisk: number;
    moonAzimuth: number;
    moonElevation: number;
    hazeDensity: number;
    hazeColor: number;
    cloudsColor: number;
}

export function sampleGradient(elevationDegrees: number): SkylabOutput {
    const el = Math.max(-90, Math.min(90, elevationDegrees));

    let lo = 0;
    let hi = STOPS.length - 1;
    for (let i = 0; i < STOPS.length - 1; i++) {
        if (el >= STOPS[i].elevation && el <= STOPS[i + 1].elevation) {
            lo = i;
            hi = i + 1;
            break;
        }
    }

    if (el <= STOPS[0].elevation) {
        lo = 0;
        hi = 0;
    } else if (el >= STOPS[STOPS.length - 1].elevation) {
        lo = STOPS.length - 1;
        hi = STOPS.length - 1;
    }

    const a = STOPS[lo];
    const b = STOPS[hi];
    const range = b.elevation - a.elevation;
    const linear = range > 0 ? (el - a.elevation) / range : 0;
    const t = linear * linear * (3 - 2 * linear);

    return {
        sunColor: packColor(lerpColor(a.sunColor, b.sunColor, t)),
        sunIntensity: lerp(a.sunIntensity, b.sunIntensity, t),
        ambientColor: packColor(lerpColor(a.ambientColor, b.ambientColor, t)),
        ambientIntensity: lerp(a.ambientIntensity, b.ambientIntensity, t),
        zenith: packColor(lerpColor(a.zenith, b.zenith, t)),
        horizon: packColor(lerpColor(a.horizon, b.horizon, t)),
        sunDisk: lerp(a.sunDisk, b.sunDisk, t),
        sunGlow: lerp(a.sunGlow, b.sunGlow, t),
        starsIntensity: lerp(a.starsIntensity, b.starsIntensity, t),
        moonDisk: lerp(a.moonDisk, b.moonDisk, t),
        moonAzimuth: 0,
        moonElevation: lerp(a.moonElevationOffset, b.moonElevationOffset, t),
        hazeDensity: lerp(a.hazeDensity, b.hazeDensity, t),
        hazeColor: packColor(lerpColor(a.hazeColor, b.hazeColor, t)),
        cloudsColor: packColor(lerpColor(a.cloudsColor, b.cloudsColor, t)),
    };
}

export function directionToElevation(dx: number, dy: number, dz: number): number {
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 0.0001) return 90;
    return Math.asin(Math.min(1, Math.max(-1, -dy / len))) * (180 / Math.PI);
}

export function directionToAzimuth(dx: number, dy: number, dz: number): number {
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 0.0001) return 0;
    return (Math.atan2(-dx / len, -dz / len) * (180 / Math.PI) + 360) % 360;
}

export function toDirection(azimuthDeg: number, elevationDeg: number): [number, number, number] {
    const az = (azimuthDeg * Math.PI) / 180;
    const el = (elevationDeg * Math.PI) / 180;
    const cosEl = Math.cos(el);
    return [-Math.sin(az) * cosEl, -Math.sin(el), -Math.cos(az) * cosEl];
}

function moonDirection(sunAzimuth: number, sunElevation: number): [number, number, number] {
    const moonAz = (sunAzimuth + 180) % 360;
    return toDirection(moonAz, Math.abs(sunElevation));
}

function lightDirection(azimuth: number, elevation: number): [number, number, number] {
    if (elevation >= 0) return toDirection(azimuth, elevation);
    if (elevation <= -18) return moonDirection(azimuth, elevation);
    const linear = -elevation / 18;
    const t = linear * linear * (3 - 2 * linear);
    const [sx, sy, sz] = toDirection(azimuth, elevation);
    const [mx, my, mz] = moonDirection(azimuth, elevation);
    const bx = sx + (mx - sx) * t;
    const by = sy + (my - sy) * t;
    const bz = sz + (mz - sz) * t;
    const len = Math.hypot(bx, by, bz);
    if (len < 0.001) return moonDirection(azimuth, elevation);
    return [bx / len, by / len, bz / len];
}

const SkylabSystem: System = {
    group: "simulation",
    annotations: { mode: "always" },

    update(state: State) {
        const eid = state.only([Skylab]);
        if (eid >= 0) {
            const azimuth = Skylab.azimuth[eid];
            const elevation = Skylab.elevation[eid];
            const [dirX, dirY, dirZ] = lightDirection(azimuth, elevation);
            const output = sampleGradient(elevation);

            const lightEid = state.only([DirectionalLight]);
            if (lightEid >= 0) {
                DirectionalLight.directionX[lightEid] = dirX;
                DirectionalLight.directionY[lightEid] = dirY;
                DirectionalLight.directionZ[lightEid] = dirZ;
                DirectionalLight.color[lightEid] = output.sunColor;
                DirectionalLight.intensity[lightEid] = output.sunIntensity;
                const shadowLinear = Math.max(0, Math.min(1, (elevation + 0.5) / 1.5));
                DirectionalLight.shadows[lightEid] =
                    shadowLinear * shadowLinear * (3 - 2 * shadowLinear);
            }

            const ambientEid = state.only([AmbientLight]);
            if (ambientEid >= 0) {
                AmbientLight.color[ambientEid] = output.ambientColor;
                AmbientLight.intensity[ambientEid] = output.ambientIntensity;
            }

            if (state.hasComponent(eid, Sky)) {
                Sky.zenith[eid] = Skylab.skyColor[eid] || output.zenith;
                Sky.horizon[eid] = Skylab.horizonColor[eid] || output.horizon;
            }

            if (state.hasComponent(eid, Sun)) {
                Sun.size[eid] = output.sunDisk;
                Sun.glow[eid] = output.sunGlow;
                Sun.azimuth[eid] = azimuth;
                Sun.elevation[eid] = elevation;
            }

            if (state.hasComponent(eid, Stars)) {
                Stars.intensity[eid] = output.starsIntensity;
            }

            if (state.hasComponent(eid, Moon)) {
                Moon.opacity[eid] = output.moonDisk;
                Moon.azimuth[eid] = (azimuth + 180) % 360;
                Moon.elevation[eid] = Math.abs(elevation);
            }

            if (state.hasComponent(eid, Haze)) {
                Haze.density[eid] = output.hazeDensity;
                Haze.color[eid] = output.hazeColor;
            }

            if (state.hasComponent(eid, Clouds)) {
                Clouds.color[eid] = output.cloudsColor;
            }
        }
    },
};

export const SkylabPlugin: Plugin = {
    name: "Skylab",
    systems: [SkylabSystem],
    components: { Skylab },
};
