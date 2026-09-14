// Seat policy: which capability a requirement tag actually names, and why a host that lacks it refuses.
// Pure and host-independent — every fact reaches these functions as data, so the same policy resolves a
// macOS run, a fixture and a host nobody has run yet. Nothing here launches, probes or reads the
// environment; `./launch` holds the per-host launch declarations and `./driver` does the observing.

import { type AdapterFacts, classifyAdapter } from "../engine/runtime/adapter";
import { type CaptureIdentity, captureIdentityLabel, captureIdentityMatches } from "./capture";
import type { LaunchPlan } from "./launch";

// Keep the harness seat module's public imports stable while the engine owns adapter policy.
export {
    type AdapterClass,
    type AdapterFacts,
    type AdapterInfoFacts,
    type AdapterVerdict,
    adapterIdentity,
    classifyAdapter,
} from "../engine/runtime/adapter";

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
