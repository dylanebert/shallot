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

export interface FlowDef {
    name: string;
    scene: string;
    viewport?: { width: number; height: number };
}
