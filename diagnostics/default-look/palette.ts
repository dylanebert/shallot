export type Rgba = readonly [number, number, number, number];

export interface LookPalette {
    clearColor: number;
    materials: {
        blue: Rgba;
        blueGround: Rgba;
        orange: Rgba;
        greyboxGround: Rgba;
        greyboxStep: Rgba;
        greyboxLift: Rgba;
        greyboxTower: Rgba;
        greyboxPerch: Rgba;
        embedded: Rgba;
    };
    textColor: number;
    ui: {
        panel: string;
        ink: string;
        border: string;
    };
}

// These are the current examples' material values, gathered here so every scene and later palette
// candidate has one content-colour source. Lighting stays the examples' warm sun and cool ambient.
export const PALETTES = {
    current: {
        clearColor: 0x1c1917,
        materials: {
            blue: [0.27, 0.58, 0.94, 1],
            blueGround: [0.19, 0.23, 0.3, 1],
            orange: [0.94, 0.58, 0.27, 1],
            greyboxGround: [0.15, 0.19, 0.2, 1],
            greyboxStep: [0.28, 0.34, 0.34, 1],
            greyboxLift: [0.83, 0.53, 0.24, 1],
            greyboxTower: [0.34, 0.4, 0.39, 1],
            greyboxPerch: [0.39, 0.45, 0.43, 1],
            embedded: [0.85, 0.55, 0.35, 0.72],
        },
        textColor: 0xffffff,
        ui: {
            panel: "rgba(14, 17, 20, 0.72)",
            ink: "#ffffff",
            border: "rgba(255, 255, 255, 0.16)",
        },
    },
} satisfies Record<string, LookPalette>;

export type PaletteName = keyof typeof PALETTES;
