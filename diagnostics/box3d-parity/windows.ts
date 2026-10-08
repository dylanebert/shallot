// Windows sized to the fewest steps whose median the run repeats: joint_grid is steady after its first
// steps; rain's last column spawns before 280; junkyard's rocks have landed and its contacts climb.
// Pyramids use the last 20 benchmark steps to measure resting stacks' contact recycle path.
export const WINDOWS: Record<string, readonly [number, number]> = {
    // biome-ignore lint/style/useNamingConvention: Box3D's benchmark name, as native.c takes it.
    joint_grid: [20, 40],
    rain: [280, 320],
    junkyard: [180, 200],
    // biome-ignore lint/style/useNamingConvention: Box3D's benchmark name.
    many_pyramids: [80, 100],
    // biome-ignore lint/style/useNamingConvention: Box3D's benchmark name.
    large_pyramid: [180, 200],
};
