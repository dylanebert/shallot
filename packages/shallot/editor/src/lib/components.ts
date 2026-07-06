import {
    Aperture,
    ArrowRight,
    Axis3d,
    Box,
    Camera,
    ChevronRight,
    CircleDashed,
    CircleDot,
    Cloud,
    Diamond,
    Eye,
    Gamepad2,
    Grid3X3,
    Image,
    Layers,
    Lightbulb,
    Minus,
    Monitor,
    Moon as MoonIcon,
    Plus,
    Scan,
    Search,
    ShieldCheck,
    SlidersHorizontal,
    Sparkles,
    Star,
    Sun,
    Sunrise,
    TriangleAlert,
    Type,
    Wind,
    Zap,
} from "lucide-static";

export type ComponentMeta = { icon: string; color: string; category?: string };

export { ChevronRight, Plus, TriangleAlert };

export type Category = { id: string; label: string; color: string };

export const CATEGORIES: Category[] = [
    { id: "spatial", label: "Spatial", color: "var(--cat-spatial)" },
    { id: "geometry", label: "Geometry", color: "var(--cat-rendering)" },
    { id: "camera", label: "Camera", color: "var(--cat-camera)" },
    { id: "lighting", label: "Lighting", color: "var(--cat-lighting)" },
    { id: "effects", label: "Effects", color: "var(--cat-effects)" },
    { id: "environment", label: "Environment", color: "var(--cat-environment)" },
    { id: "pipeline", label: "Pipeline", color: "var(--cat-pipeline)" },
    { id: "drawing", label: "Drawing", color: "var(--cat-drawing)" },
    { id: "gameplay", label: "Gameplay", color: "var(--cat-gameplay)" },
];

const CATEGORY_MAP = new Map(CATEGORIES.map((c) => [c.id, c]));

export function getCategory(id: string): Category | undefined {
    return CATEGORY_MAP.get(id);
}

const META: Record<string, ComponentMeta> = {
    transform: { icon: Axis3d, color: "var(--cat-spatial)", category: "spatial" },
    "world-transform": { icon: Axis3d, color: "var(--cat-spatial)", category: "spatial" },

    part: { icon: Box, color: "var(--cat-rendering)", category: "geometry" },
    surface: { icon: Layers, color: "var(--cat-rendering)", category: "geometry" },
    canvas: { icon: Image, color: "var(--cat-rendering)", category: "geometry" },

    camera: { icon: Camera, color: "var(--cat-camera)", category: "camera" },
    orbit: { icon: CircleDot, color: "var(--cat-camera)", category: "camera" },
    viewport: { icon: Monitor, color: "var(--cat-camera)", category: "camera" },

    "ambient-light": { icon: Sun, color: "var(--cat-lighting)", category: "lighting" },
    "directional-light": { icon: Sunrise, color: "var(--cat-lighting)", category: "lighting" },
    "point-light": { icon: Lightbulb, color: "var(--cat-lighting)", category: "lighting" },
    shadows: { icon: ShieldCheck, color: "var(--cat-lighting)", category: "lighting" },
    tonemap: { icon: SlidersHorizontal, color: "var(--cat-effects)", category: "effects" },
    fxaa: { icon: Scan, color: "var(--cat-effects)", category: "effects" },
    vignette: { icon: Aperture, color: "var(--cat-effects)", category: "effects" },
    bloom: { icon: Sparkles, color: "var(--cat-effects)", category: "effects" },
    posterize: { icon: Grid3X3, color: "var(--cat-effects)", category: "effects" },
    dither: { icon: CircleDashed, color: "var(--cat-effects)", category: "effects" },
    volumetrics: { icon: Eye, color: "var(--cat-effects)", category: "effects" },
    reflections: { icon: Eye, color: "var(--cat-effects)", category: "effects" },

    sky: { icon: Cloud, color: "var(--cat-environment)", category: "environment" },
    sun: { icon: Sun, color: "var(--cat-environment)", category: "environment" },
    moon: { icon: MoonIcon, color: "var(--cat-environment)", category: "environment" },
    stars: { icon: Star, color: "var(--cat-environment)", category: "environment" },
    clouds: { icon: Cloud, color: "var(--cat-environment)", category: "environment" },
    haze: { icon: Wind, color: "var(--cat-environment)", category: "environment" },
    skylab: { icon: Sunrise, color: "var(--cat-environment)", category: "environment" },

    texel: { icon: Zap, color: "var(--cat-pipeline)", category: "pipeline" },
    raytracing: { icon: Search, color: "var(--cat-pipeline)", category: "pipeline" },
    dynamic: { icon: Zap, color: "var(--cat-pipeline)", category: "pipeline" },

    line: { icon: Minus, color: "var(--cat-drawing)", category: "drawing" },
    arrow: { icon: ArrowRight, color: "var(--cat-drawing)", category: "drawing" },
    text: { icon: Type, color: "var(--cat-drawing)", category: "drawing" },

    player: { icon: Gamepad2, color: "var(--cat-gameplay)", category: "gameplay" },
    gizmos: { icon: Grid3X3, color: "var(--cat-gameplay)", category: "gameplay" },
};

const DEFAULT_META: ComponentMeta = { icon: Diamond, color: "var(--accent)" };

export function getMeta(name: string): ComponentMeta {
    return META[name] ?? DEFAULT_META;
}

export type ComponentGroup = { category: Category; items: string[] };

export function groupComponents(names: string[]): ComponentGroup[] {
    const buckets = new Map<string, string[]>();
    const uncategorized: string[] = [];

    for (const name of names) {
        const meta = META[name];
        if (meta?.category) {
            let bucket = buckets.get(meta.category);
            if (!bucket) {
                bucket = [];
                buckets.set(meta.category, bucket);
            }
            bucket.push(name);
        } else {
            uncategorized.push(name);
        }
    }

    const groups: ComponentGroup[] = [];
    for (const cat of CATEGORIES) {
        const items = buckets.get(cat.id);
        if (items && items.length > 0) groups.push({ category: cat, items });
    }

    if (uncategorized.length > 0) {
        groups.push({
            category: { id: "other", label: "Other", color: "var(--accent)" },
            items: uncategorized,
        });
    }

    return groups;
}

const HERO_PRIORITY: string[] = [
    "camera",
    "directional-light",
    "ambient-light",
    "point-light",
    "part",
    "surface",
];

export function heroMeta(attrs: { name: string }[]): ComponentMeta {
    for (const hero of HERO_PRIORITY) {
        if (attrs.some((a) => a.name === hero)) return META[hero];
    }
    return DEFAULT_META;
}

/** the component a node is best identified by: its hero type, else its first component */
export function heroName(attrs: { name: string }[]): string | undefined {
    for (const hero of HERO_PRIORITY) {
        if (attrs.some((a) => a.name === hero)) return hero;
    }
    return attrs[0]?.name;
}

/** how a node titles in the outliner + inspector: its id, else its type, else a bare "entity" */
export function nodeLabel(node: { id?: string; attrs: { name: string }[] }): string {
    return node.id || heroName(node.attrs) || "entity";
}
