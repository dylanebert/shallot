/** the four adapter cases a seat has to tell apart. Only `real` can carry a positive device claim. */
export type AdapterClass = "absent" | "fallback" | "unidentified" | "real";

/** the adapter identity fields WebGPU exposes, as read from `GPUAdapter.info`. */
export interface AdapterInfoFacts {
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
    /** WebGPU's own fallback flag, when the browser reports it. */
    isFallbackAdapter?: boolean;
}

/** what an observer saw when it asked for an adapter. `present: false` is a genuinely absent adapter. */
export interface AdapterFacts {
    present: boolean;
    info?: AdapterInfoFacts;
}

/** one adapter classification: its case, the identity string that travels with a verdict, and the
 * refusal reason for every case but `real`. */
export interface AdapterVerdict {
    class: AdapterClass;
    identity: string;
    reason?: string;
}

// Chromium's software path names itself in the adapter identity: SwiftShader on every platform, and the
// other common software rasterizers a host may fall back to. Matching the identity is what catches a
// software adapter whose `isFallbackAdapter` reads false because the browser selected it as the only
// adapter rather than as the requested fallback.
const SOFTWARE_MARKERS = [
    "swiftshader",
    "llvmpipe",
    "lavapipe",
    "softpipe",
    "basic render",
    "software adapter",
    "software rasterizer",
    "microsoft basic",
] as const;

function identityParts(info: AdapterInfoFacts | undefined): string[] {
    return [info?.vendor, info?.architecture, info?.device, info?.description]
        .map((part) => (typeof part === "string" ? part.trim() : ""))
        .filter((part) => part !== "");
}

/** the adapter identity that travels with a verdict, or `unidentified` when the browser masked it all. */
export function adapterIdentity(info: AdapterInfoFacts | undefined): string {
    const parts = identityParts(info);
    return parts.length === 0 ? "unidentified" : parts.join(" ");
}

/** Classify an adapter. A masked identity is `unidentified`, never a real-device claim. */
export function classifyAdapter(facts: AdapterFacts): AdapterVerdict {
    const identity = adapterIdentity(facts.info);
    if (!facts.present) {
        return { class: "absent", identity: "none", reason: "no WebGPU adapter is available" };
    }
    const marker = SOFTWARE_MARKERS.find((name) => identity.toLowerCase().includes(name));
    if (facts.info?.isFallbackAdapter === true || marker !== undefined) {
        return {
            class: "fallback",
            identity,
            reason: `WebGPU reports a fallback adapter, not a real device: ${identity}`,
        };
    }
    if (identity === "unidentified") {
        return {
            class: "unidentified",
            identity,
            reason: "WebGPU reports an adapter with no identity, so no real device can be claimed",
        };
    }
    return { class: "real", identity };
}
