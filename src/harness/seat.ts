// Seat policy: which capability a requirement tag actually names, and why a host that lacks it refuses.
// Pure and host-independent — every fact reaches these functions as data, so the same policy resolves a
// macOS run, a fixture and a host nobody has run yet. Nothing here launches, probes or reads the
// environment; `./launch` holds the per-host launch declarations and `./driver` does the observing.

import { type CaptureIdentity, captureIdentityLabel, captureIdentityMatches } from "./capture";
import type { LaunchPlan } from "./launch";

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
 *  refusal reason for every case but `real`. */
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

/** the seats a requirement tag names. `cpu` is the absence of a requirement, not a tag. */
export const SEATS = ["cpu", "gpu", "chromium", "display"] as const;
export type Seat = (typeof SEATS)[number];

/** the capture geometry an observer actually saw, against which the contract is checked. */
export interface CaptureFacts {
    identity: CaptureIdentity;
}

/** the browser seat's observed facts: how it was launched, what adapter it reached, what it captured. */
export interface BrowserFacts {
    launch?: LaunchPlan;
    adapter?: AdapterFacts;
    capture?: CaptureFacts;
}

/** a genuinely headed premise: a real display or a physical-interaction seat, named by its source. */
export interface DisplayFacts {
    headed: boolean;
    source: string;
}

/**
 * every fact a seat resolution may read. Each seat reads only its own field, which is what keeps one seat
 * from silently standing in for another: a real `device` adapter cannot satisfy `chromium`, and a headless
 * browser cannot satisfy `display`.
 */
export interface SeatFacts {
    /** the in-process (Bun) WebGPU adapter, for the `gpu` seat. */
    device?: AdapterFacts;
    /** the browser-composited seat, for `chromium`. */
    browser?: BrowserFacts;
    /** the headed premise, for `display`. */
    display?: DisplayFacts;
}

/** a seat resolution: available, or refused with the reason a verdict prints. */
export type SeatResolution = { ok: true; detail: string } | { ok: false; reason: string };

function refuse(seat: Seat, reason: string): SeatResolution {
    return { ok: false, reason: `${seat} seat unavailable: ${reason}` };
}

/**
 * Resolve one seat from observed facts.
 *
 * - `cpu` — no requirement; always available.
 * - `gpu` — a real in-process WebGPU device. Reads no browser fact, so a browser claim never grants it.
 * - `chromium` — a declared headless launch, a positively identified real adapter inside that browser, and
 *   a capture at the one declared identity.
 * - `display` — a genuinely headed display or physical-interaction premise; headless never grants it.
 *
 * @example const seat = resolveSeat("gpu", { device: { present: true, info } });
 */
export function resolveSeat(
    seat: Seat,
    facts: SeatFacts,
    contract?: CaptureIdentity,
): SeatResolution {
    if (seat === "cpu") return { ok: true, detail: "cpu" };
    if (seat === "gpu") {
        if (facts.device === undefined)
            return refuse(seat, "no in-process WebGPU device was probed");
        const adapter = classifyAdapter(facts.device);
        if (adapter.class !== "real") return refuse(seat, adapter.reason ?? adapter.class);
        return { ok: true, detail: `real device ${adapter.identity}` };
    }
    if (seat === "display") {
        if (facts.display === undefined || !facts.display.headed) {
            return refuse(seat, "no headed display or physical-interaction premise is declared");
        }
        return { ok: true, detail: `headed display via ${facts.display.source}` };
    }
    const browser = facts.browser;
    if (browser?.launch === undefined) {
        return refuse(seat, "no declared headless Chromium launch path for this host");
    }
    if (browser.adapter === undefined) {
        return refuse(seat, "the browser reported no adapter observation");
    }
    const adapter = classifyAdapter(browser.adapter);
    if (adapter.class !== "real") return refuse(seat, adapter.reason ?? adapter.class);
    if (contract !== undefined) {
        if (browser.capture === undefined) {
            return refuse(seat, "the browser reported no capture at the declared capture contract");
        }
        if (!captureIdentityMatches(browser.capture.identity, contract)) {
            return refuse(
                seat,
                `capture identity ${captureIdentityLabel(browser.capture.identity)} is not the declared contract ${captureIdentityLabel(contract)}`,
            );
        }
    }
    return {
        ok: true,
        detail: `${browser.launch.mode} chromium on real adapter ${adapter.identity}`,
    };
}
