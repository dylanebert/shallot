import { DUSK_SKY_DEFAULTS } from "./sky";

/** Sun-facing capture condition measured by the solar-energy claim. */
export const SUN_FACING = {
    name: "sun-facing",
    camera: { yaw: Math.PI + 0.35, pitch: 0 },
} as const;

/** matched showcase capture conditions. */
export const CAPTURE = {
    width: 1280,
    height: 720,
    time: 6,
    camera: { distance: 18, yaw: 0, pitch: 0.24 },
    sunAzimuthOffset: 0.35,
    sunElevation: 0.32,
    sky: DUSK_SKY_DEFAULTS,
} as const;

/** light-travel direction derived from the declared sun angles. */
export function sunDirection(azimuth: number, elevation: number): [number, number, number] {
    const horizontal = Math.cos(elevation);
    return [-horizontal * Math.sin(azimuth), -Math.sin(elevation), -horizontal * Math.cos(azimuth)];
}
