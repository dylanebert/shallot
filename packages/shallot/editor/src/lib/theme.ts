// The editor's single source of color. Every UI surface, the loading spinner, and the themed viewport
// overlays (grid, selection outline) read from one Palette. A theme is one object
// satisfying Palette; switching is a swap (setTheme) — CSS cascades from the applied custom
// properties, and the GPU/canvas code reads `current.palette` each frame, so a switch retints the
// whole editor live with no rebuild. No color literal lives anywhere else in the editor: CSS tints
// derive from these tokens via `color-mix`, GPU colors via `packed` / `rgb`.

/** the editor color tokens — one theme is one object satisfying this interface */
export interface Palette {
    /** background polarity — sets the CSS `color-scheme` (native controls) and is the cue a light theme
     * derives its chromatic tokens for a light background (see `adaptLight`) */
    mode: "light" | "dark";
    /** app background — the base surface */
    bg: string;
    /** opaque surfaces, stepping away from bg; 1 = panels, 3 = floating menus/popovers (see editor-ui.md) */
    surface1Solid: string;
    surface2Solid: string;
    surface3Solid: string;
    surface4Solid: string;
    /** translucent elevation overlays (tint matches the theme's temperature), 1 = subtlest */
    surface1: string;
    surface2: string;
    surface3: string;
    surface4: string;
    /** divider / outline */
    border: string;
    /** primary / secondary / muted text */
    text: string;
    textSecondary: string;
    textMuted: string;
    /** shallot gold accent + its hover tint */
    accent: string;
    accentHover: string;
    /** status hues */
    error: string;
    warning: string;
    /** component-category hues */
    catSpatial: string;
    catRendering: string;
    catLighting: string;
    catCamera: string;
    catEffects: string;
    catEnvironment: string;
    catPipeline: string;
    catDrawing: string;
    catGameplay: string;
    /** selection outline */
    outline: string;
    /** ground-grid line color */
    grid: string;
    /** loading spinner track */
    spinnerTrack: string;
    /** grid + transform-gizmo axes — functional cues (red X / green Y / blue Z) */
    axisX: string;
    axisY: string;
    axisZ: string;
}

/** Shallot — the default. Warm, opinionated surfaces (gruvbox-derived), shallot gold accent */
const shallot: Palette = {
    mode: "dark",
    bg: "#1d1a17",
    surface1Solid: "#282421",
    surface2Solid: "#322c28",
    surface3Solid: "#3c352f",
    surface4Solid: "#463d35",
    surface1: "rgba(235, 219, 178, 0.04)",
    surface2: "rgba(235, 219, 178, 0.08)",
    surface3: "rgba(235, 219, 178, 0.13)",
    surface4: "rgba(235, 219, 178, 0.18)",
    border: "rgba(235, 219, 178, 0.1)",
    text: "#ebdbb2",
    textSecondary: "#c8b89a",
    textMuted: "#a89984",
    accent: "#d49560",
    accentHover: "#e8a86b",
    error: "#fb4934",
    warning: "#fabd2f",
    catSpatial: "#4a90e2",
    catRendering: "#50c878",
    catLighting: "#fbbf24",
    catCamera: "#5bb8c4",
    catEffects: "#a78bda",
    catEnvironment: "#6cb4d9",
    catPipeline: "#e0915c",
    catDrawing: "#d4708f",
    catGameplay: "#7bc876",
    outline: "#ff6a00",
    grid: "#d1ccc2",
    spinnerTrack: "#3c352f",
    axisX: "#cc3333",
    axisY: "#3aa83a",
    axisZ: "#3366cc",
};

/** Modern Dark — whisper-warm neutral surfaces, hardened text, shallot gold accent */
const dark: Palette = {
    mode: "dark",
    bg: "#1a1917",
    surface1Solid: "#211f1d",
    surface2Solid: "#292724",
    surface3Solid: "#322f2c",
    surface4Solid: "#3d3a36",
    surface1: "rgba(240, 232, 222, 0.04)",
    surface2: "rgba(240, 232, 222, 0.07)",
    surface3: "rgba(240, 232, 222, 0.11)",
    surface4: "rgba(240, 232, 222, 0.16)",
    border: "rgba(240, 232, 222, 0.1)",
    text: "#ece8e3",
    textSecondary: "#c4beb6",
    textMuted: "#a39b92",
    accent: "#d49560",
    accentHover: "#e8a86b",
    error: "#e8605a",
    warning: "#e0a52b",
    catSpatial: "#4a90e2",
    catRendering: "#50c878",
    catLighting: "#fbbf24",
    catCamera: "#5bb8c4",
    catEffects: "#a78bda",
    catEnvironment: "#6cb4d9",
    catPipeline: "#e0915c",
    catDrawing: "#d4708f",
    catGameplay: "#7bc876",
    outline: "#ff6a00",
    grid: "#c9c5bf",
    spinnerTrack: "#322f2c",
    axisX: "#cc3333",
    axisY: "#3aa83a",
    axisZ: "#3366cc",
};

/** Modern Light — VS Code light modern, shallot-accented. Also the dogfood check that every element
 * reads its color from a token (an untokenized literal stays dark and stands out on a light surface) */
const light: Palette = {
    mode: "light",
    // surfaces are VS Code "Light Modern" verbatim: #f8f8f8 sidebars/panels, #ffffff editor/menus,
    // #e5e5e5 borders. Overlays are neutral grey (VS Code's hovers carry no tint)
    bg: "#f3f3f3",
    surface1Solid: "#f8f8f8",
    surface2Solid: "#fbfbfb",
    surface3Solid: "#ffffff",
    surface4Solid: "#ffffff",
    surface1: "rgba(0, 0, 0, 0.04)",
    surface2: "rgba(0, 0, 0, 0.06)",
    surface3: "rgba(0, 0, 0, 0.09)",
    surface4: "rgba(0, 0, 0, 0.13)",
    border: "#e5e5e5",
    text: "#3b3b3b",
    textSecondary: "#595959",
    textMuted: "#6e6e6e",
    // chromatic tokens are the dark theme's brand/category hues, re-lit into VS Code's accent space
    // (see adaptLight). hover darkens on light (vs lightens on dark)
    accent: adaptLight(dark.accent),
    accentHover: darken(adaptLight(dark.accent), 0.05),
    error: adaptLight(dark.error),
    warning: adaptLight(dark.warning),
    catSpatial: adaptLight(dark.catSpatial),
    catRendering: adaptLight(dark.catRendering),
    catLighting: adaptLight(dark.catLighting),
    catCamera: adaptLight(dark.catCamera),
    catEffects: adaptLight(dark.catEffects),
    catEnvironment: adaptLight(dark.catEnvironment),
    catPipeline: adaptLight(dark.catPipeline),
    catDrawing: adaptLight(dark.catDrawing),
    catGameplay: adaptLight(dark.catGameplay),
    outline: "#ff6a00",
    grid: "#909090",
    spinnerTrack: "#e0e0e0",
    axisX: "#cc3333",
    axisY: "#3aa83a",
    axisZ: "#3366cc",
};

/** Neutral — fully achromatic chrome for color-critical work. Saturated UI biases viewport color
 * perception (chromatic adaptation), so the accent, categories, and grid are gray and the
 * selection outline is white — the scene's colors are judged with no editor tint. Status hues stay
 * mildly desaturated so errors/warnings remain legible */
const neutral: Palette = {
    mode: "dark",
    bg: "#1a1a1a",
    surface1Solid: "#212121",
    surface2Solid: "#2a2a2a",
    surface3Solid: "#333333",
    surface4Solid: "#3d3d3d",
    surface1: "rgba(255, 255, 255, 0.04)",
    surface2: "rgba(255, 255, 255, 0.07)",
    surface3: "rgba(255, 255, 255, 0.11)",
    surface4: "rgba(255, 255, 255, 0.16)",
    border: "rgba(255, 255, 255, 0.1)",
    text: "#e8e8e8",
    textSecondary: "#b8b8b8",
    textMuted: "#8c8c8c",
    accent: "#b0b0b0",
    accentHover: "#cfcfcf",
    error: "#c77b76",
    warning: "#b9985a",
    catSpatial: "#9a9a9a",
    catRendering: "#888888",
    catLighting: "#b5b5b5",
    catCamera: "#a0a0a0",
    catEffects: "#787878",
    catEnvironment: "#909090",
    catPipeline: "#ababab",
    catDrawing: "#828282",
    catGameplay: "#989898",
    outline: "#ffffff",
    grid: "#b0b0b0",
    spinnerTrack: "#333333",
    axisX: "#cc3333",
    axisY: "#3aa83a",
    axisZ: "#3366cc",
};

/** a selectable theme — id is persisted in prefs, label shows in the menu */
export interface Theme {
    id: string;
    label: string;
    palette: Palette;
}

/** the theme registry — the menu and the prefs both key off this; first entry is the default */
export const THEMES: Theme[] = [
    { id: "shallot", label: "Shallot", palette: shallot },
    { id: "dark", label: "Dark", palette: dark },
    { id: "light", label: "Light", palette: light },
    { id: "neutral", label: "Neutral", palette: neutral },
];

/** the live palette read by GPU/canvas code each frame (grid uniform, loading spinner). Mutated by
 * {@link setTheme}, so a switch retints the viewport without a rebuild */
export const current: { palette: Palette } = { palette: THEMES[0].palette };

// the CSS-referenced tokens and their custom-property names. The viewport backdrop tokens are
// absent — they're read in TS (packed / rgb), never as CSS vars.
const CSS_VARS: [keyof Palette, string][] = [
    ["bg", "--bg"],
    ["surface1", "--surface-1"],
    ["surface2", "--surface-2"],
    ["surface3", "--surface-3"],
    ["surface4", "--surface-4"],
    ["surface1Solid", "--surface-1-solid"],
    ["surface2Solid", "--surface-2-solid"],
    ["surface3Solid", "--surface-3-solid"],
    ["surface4Solid", "--surface-4-solid"],
    ["border", "--border"],
    ["text", "--text"],
    ["textSecondary", "--text-secondary"],
    ["textMuted", "--text-muted"],
    ["accent", "--accent"],
    ["accentHover", "--accent-hover"],
    ["error", "--error"],
    ["warning", "--warning"],
    ["catSpatial", "--cat-spatial"],
    ["catRendering", "--cat-rendering"],
    ["catLighting", "--cat-lighting"],
    ["catCamera", "--cat-camera"],
    ["catEffects", "--cat-effects"],
    ["catEnvironment", "--cat-environment"],
    ["catPipeline", "--cat-pipeline"],
    ["catDrawing", "--cat-drawing"],
    ["catGameplay", "--cat-gameplay"],
];

/** resolve a theme by id, falling back to the default */
export function theme(id: string | undefined): Theme {
    return THEMES.find((t) => t.id === id) ?? THEMES[0];
}

/** select a theme: update the live palette and write its tokens onto the root as CSS custom
 * properties. The viewport backdrop (clear/outline) is updated by the caller from `current.palette` */
export function setTheme(root: HTMLElement, id: string): void {
    const p = theme(id).palette;
    current.palette = p;
    // native controls (scrollbars, select popups) follow color-scheme, so they match a light theme
    root.style.colorScheme = p.mode;
    for (const [key, name] of CSS_VARS) root.style.setProperty(name, p[key]);
}

/** `"#221e1a"` → `0x221e1a`, for GPU clear colors / packed-color fields */
export function packed(hex: string): number {
    return parseInt(hex.slice(1), 16);
}

/** `"#c9c5bf"` → normalized `[r, g, b]` in 0..1, for shader uniform colors */
export function rgb(hex: string): [number, number, number] {
    const n = packed(hex);
    return [((n >> 16) & 0xff) / 255, ((n >> 8) & 0xff) / 255, (n & 0xff) / 255];
}

// sRGB ⇄ OKLab/OKLCH (Ottosson). OKLCH separates lightness (L) from hue (H) + chroma (C), so a color
// can be re-lit for a different background polarity without shifting its hue — the basis for adaptLight.
function srgbToLinear(c: number): number {
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(c: number): number {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
}

function oklch(hex: string): [number, number, number] {
    const [r, g, b] = rgb(hex).map(srgbToLinear) as [number, number, number];
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    const lab0 = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
    const laba = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
    const labb = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
    return [lab0, Math.hypot(laba, labb), Math.atan2(labb, laba)];
}

function oklchHex(lightness: number, chroma: number, hue: number): string {
    const a = chroma * Math.cos(hue);
    const b = chroma * Math.sin(hue);
    const l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const s = (lightness - 0.0894841775 * a - 1.291485548 * b) ** 3;
    const lin = [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
    ];
    const byte = (c: number) =>
        Math.max(0, Math.min(255, Math.round(linearToSrgb(c) * 255)))
            .toString(16)
            .padStart(2, "0");
    return `#${byte(lin[0])}${byte(lin[1])}${byte(lin[2])}`;
}

/** re-light a chromatic token authored for a dark background so it reads on a light one. VS Code "Light
 * Modern" sits its accent (#005fb8) at OKLCH L≈0.49, C≈0.16; the source hue is rendered at that lightness
 * with chroma capped near it, so light-theme chromatic tokens share VS Code's proven, readable space —
 * only the hue is the token's own, so a light theme inherits the brand/category hues, not hand-copies */
export function adaptLight(hex: string): string {
    const [, chroma, hue] = oklch(hex);
    return oklchHex(0.49, Math.min(chroma, 0.184), hue);
}

function darken(hex: string, amount: number): string {
    const [l, c, h] = oklch(hex);
    return oklchHex(l - amount, c, h);
}
