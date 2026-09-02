/** matched showcase capture conditions. */
export const CAPTURE = {
    width: 1280,
    height: 720,
    time: 6,
    camera: { distance: 18, yaw: 0, pitch: 0.24 },
    sunAzimuthOffset: 0.35,
    sunElevation: 0.32,
} as const;

/** light-travel direction derived from the declared sun angles. */
export function sunDirection(azimuth: number, elevation: number): [number, number, number] {
    const horizontal = Math.cos(elevation);
    return [-horizontal * Math.sin(azimuth), -Math.sin(elevation), -horizontal * Math.cos(azimuth)];
}
