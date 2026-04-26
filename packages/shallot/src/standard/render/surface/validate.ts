import type { SurfaceData } from "./index";

const INSTANCE_DATA_RE = /\bentityIds\b/;
const DISCARD_RE = /\bdiscard\b/;
const RETURN_RE = /\breturn\b/;
const INST_ACCESS_RE = /\binst\./;

export function validateSurface(data: SurfaceData): void {
    if (data.fragment && INSTANCE_DATA_RE.test(data.fragment)) {
        console.warn("[surface] fragment references entityIds — not available in RT pipeline");
    }

    if (data.fragment && DISCARD_RE.test(data.fragment)) {
        console.warn(
            "[surface] fragment uses 'discard' — set surface.opacity = 0.0 instead; pipeline handles discard",
        );
    }

    if (data.vertex && RETURN_RE.test(data.vertex)) {
        console.warn("[surface] vertex snippet uses 'return' — modify 'pos' variable instead");
    }

    const usesInst =
        (data.fragment && INST_ACCESS_RE.test(data.fragment)) ||
        (data.vertex && INST_ACCESS_RE.test(data.vertex));
    if (usesInst && (!data.properties || data.properties.length === 0)) {
        console.warn("[surface] snippet references 'inst.' but no properties declared");
    }
}
