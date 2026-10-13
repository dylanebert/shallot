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
const current = {
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

export type Oklch = readonly [number, number, number];
export type ColorRole = "background" | "foreground" | "accent";

export function linearFromOklch([l, c, hue]: Oklch): Rgba {
    const a = c * Math.cos((hue * Math.PI) / 180);
    const b = c * Math.sin((hue * Math.PI) / 180);
    const ll = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const s = (l - 0.0894841775 * a - 1.291485548 * b) ** 3;
    return [
        4.0767416621 * ll - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * ll + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * ll - 0.7034186147 * m + 1.707614701 * s,
        1,
    ];
}

export function srgbChannel(linear: number): number {
    return 255 * (linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055);
}

export function colorHex(linear: Rgba): string {
    return `#${linear
        .slice(0, 3)
        .map((value) => Math.round(srgbChannel(value)).toString(16).padStart(2, "0"))
        .join("")}`;
}

export function oklchFromLinear([r, g, b]: Rgba): Oklch {
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    const a = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
    const bb = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
    return [
        0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
        Math.hypot(a, bb),
        ((Math.atan2(bb, a) * 180) / Math.PI + 360) % 360,
    ];
}

export const MATERIAL_ROLES = {
    blue: "accent",
    blueGround: "foreground",
    orange: "accent",
    greyboxGround: "foreground",
    greyboxStep: "foreground",
    greyboxLift: "accent",
    greyboxTower: "background",
    greyboxPerch: "foreground",
    embedded: "accent",
} satisfies Record<keyof LookPalette["materials"], ColorRole>;

// All candidates share lightness and chroma; only the hues change.
export const CANDIDATES = {
    "dev-orange": { neutral: 80, accents: [250, 55, 100, 25] },
    "tidal-teal": { neutral: 220, accents: [290, 185, 225, 145] },
    "orchid-garden": { neutral: 320, accents: [145, 325, 300, 80] },
} as const;

export function candidateColors(hues: (typeof CANDIDATES)[keyof typeof CANDIDATES]) {
    return {
        ground: [0.7, 0.025, hues.neutral],
        dark: [0.335, 0.002, hues.neutral],
        ink: [0.92, 0.03, hues.neutral],
        primary: [0.7, 0.1, hues.accents[0]],
        secondary: [0.7, 0.1, hues.accents[1]],
        lift: [0.7, 0.1, hues.accents[2]],
        embedded: [0.7, 0.1, hues.accents[3]],
    } satisfies Record<string, Oklch>;
}

function candidate(hues: (typeof CANDIDATES)[keyof typeof CANDIDATES]): LookPalette {
    const colors = candidateColors(hues);
    const ground = linearFromOklch(colors.ground);
    const dark = linearFromOklch(colors.dark);
    const ink = colorHex(linearFromOklch(colors.ink));
    const [r, g, b] = linearFromOklch(colors.embedded);
    return {
        clearColor: 0x1c1917,
        materials: {
            blue: linearFromOklch(colors.primary),
            blueGround: ground,
            orange: linearFromOklch(colors.secondary),
            greyboxGround: ground,
            greyboxStep: ground,
            greyboxLift: linearFromOklch(colors.lift),
            greyboxTower: dark,
            greyboxPerch: ground,
            embedded: [r, g, b, 0.72],
        },
        textColor: Number.parseInt(ink.slice(1), 16),
        ui: { panel: colorHex(dark), ink, border: colorHex(ground) },
    };
}

export const PALETTES = {
    ...current,
    "dev-orange": candidate(CANDIDATES["dev-orange"]),
    "tidal-teal": candidate(CANDIDATES["tidal-teal"]),
    "orchid-garden": candidate(CANDIDATES["orchid-garden"]),
};

export type PaletteName = keyof typeof PALETTES;
