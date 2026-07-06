export interface Bounds {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface Annotation {
    type: "click" | "region" | "label";
    selector: string;
    bounds: Bounds;
    label?: string;
}

export interface Step {
    id: string;
    screenshot: string;
    timestamp: number;
    viewport: { width: number; height: number };
    annotations: Annotation[];
}

export interface Manifest {
    flow: string;
    scene: string;
    timestamp: string;
    viewport: { width: number; height: number };
    steps: Step[];
}

export interface StepOpts {
    highlight?: string | string[];
    click?: string;
    labels?: Record<string, string>;
    clip?: string;
}

/** fractional 0..1 box over the canvas, for {@link SampleOpts.region} */
export interface SampleRegion {
    x: number;
    y: number;
    w: number;
    h: number;
}

export interface SampleOpts {
    /** region to sample, fractional 0..1; defaults to the center 40% */
    region?: SampleRegion;
    /** per-channel delta past which a pixel counts as non-background (default 24) */
    bgTol?: number;
    /** an rgb to count separately — the selection outline accent (`0xff6a00` = `[255, 106, 0]`) */
    accent?: [number, number, number];
    /** per-channel tolerance for the accent match (default 40) */
    accentTol?: number;
}

export interface SampleResult {
    /** fraction of region pixels differing from the top-left corner (the clear color) */
    nonBackground: number;
    /** fraction of region pixels within `accentTol` of `accent` (0 when no accent passed) */
    accent: number;
    /** the corner pixel used as the background reference */
    background: [number, number, number];
    /** pixels sampled */
    total: number;
}

export interface FlowDef {
    name: string;
    scene: string;
    viewport?: { width: number; height: number };
    /**
     * which page the flow drives. `"editor"` (default) boots the shallot editor against the fixture;
     * `"app"` boots the standalone `run()` fixture on its own dev server (CAPTURE_APP_PORT) — the path
     * for engine features that aren't editor-shaped, like survive-reload. `"manual"` does no initial
     * navigation — the flow drives its own (`ctx.openEditor(port)`), the path for a sweep across
     * several editor servers (the zoo specimens) in one session.
     */
    target?: "editor" | "app" | "manual";
    /**
     * per-test timeout override (ms). Defaults to the config's 15s, right for a single-page flow; a
     * `manual` sweep that drives several editor servers in one test sizes this to the specimen count.
     */
    timeout?: number;
}
