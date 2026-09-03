/** matched showcase capture conditions. */
export const CAPTURE = {
    width: 1280,
    height: 720,
    time: 6,
    camera: { distance: 18, yaw: 0, pitch: 0.24 },
    sunAzimuthOffset: 0.35,
    sunElevation: 0.32,
    sky: {
        zenith: 0x31577e,
        horizon: 0xd5a6a0,
        haze: 0xe8b5a4,
        hazeStrength: 0.16,
        cloud: 0x715f78,
        cloudStrength: 0.18,
        sun: 0xffd5a0,
        sunStrength: 0.42,
        exposure: 1.05,
    },
} as const;

/** light-travel direction derived from the declared sun angles. */
export function sunDirection(azimuth: number, elevation: number): [number, number, number] {
    const horizontal = Math.cos(elevation);
    return [-horizontal * Math.sin(azimuth), -Math.sin(elevation), -horizontal * Math.cos(azimuth)];
}
