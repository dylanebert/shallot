// Host launch declarations, kept separate from seat policy. A host declares only evidence: whether a real
// adapter has actually been observed there in each launch mode. The launch mode is policy and lives here in
// code, keyed by seat, so no declaration can ask for a headed browser or change what a seat means.

import type { RealGpuLaunch } from "./browser";
import launchFloor from "./browser.json" with { type: "json" };
import hosts from "./launch.json" with { type: "json" };

/** the browser seats and the one launch mode each may use. `chromium` is always headless: a host that
 *  cannot reach a real adapter headlessly refuses it, and a window never substitutes for a headless run.
 *  Only `display`, a declared headed premise, launches headed. */
export const LAUNCH_MODES = { chromium: "headless", display: "headed" } as const;
export type LaunchSeat = keyof typeof LAUNCH_MODES;
export type LaunchMode = (typeof LAUNCH_MODES)[LaunchSeat];

/** whether a real adapter has actually been observed on a host, or is only declared. `unproven` never
 *  grants a seat — the driver still has to observe a real adapter at run time. */
export type AdapterEvidence = "proven" | "unproven";

/** one host's launch declaration: pure data, with no launch mode or capability of its own. */
export interface HostLaunchDeclaration {
    adapterEvidence: Readonly<Record<LaunchMode, AdapterEvidence>>;
    note: string;
}

/** the declared hosts, keyed by `process.platform`. An absent host has no launch path and refuses. */
export const HOST_LAUNCHES: Readonly<Record<string, HostLaunchDeclaration>> = hosts as Readonly<
    Record<string, HostLaunchDeclaration>
>;

/** the real-GPU launch floor every host and seat shares. */
export const LAUNCH_FLOOR: RealGpuLaunch = launchFloor as RealGpuLaunch;

/** a resolved launch: one seat's fixed mode plus the shared floor, for one declared host. */
export interface LaunchPlan {
    host: string;
    seat: LaunchSeat;
    mode: LaunchMode;
    channel: RealGpuLaunch["channel"];
    args: readonly string[];
    /** the host's evidence in this seat's mode. */
    adapterEvidence: AdapterEvidence;
}

/** Resolve a host's launch plan for a browser seat, or refuse a host with no declaration. */
export function launchPlan(
    host: string,
    seat: LaunchSeat = "chromium",
): LaunchPlan | { refused: string } {
    const mode = LAUNCH_MODES[seat];
    const declaration = HOST_LAUNCHES[host];
    if (declaration === undefined) {
        return {
            refused: `no declared ${mode} Chromium launch path for host ${host}; declared hosts are ${Object.keys(HOST_LAUNCHES).sort().join(", ")}`,
        };
    }
    return {
        host,
        seat,
        mode,
        channel: LAUNCH_FLOOR.channel,
        args: LAUNCH_FLOOR.args,
        adapterEvidence: declaration.adapterEvidence[mode],
    };
}

/** the Playwright launch options for a plan. `headless` comes from the seat's policy, never from host
 *  data or the plan's own fields. */
export function launchOptions(plan: LaunchPlan): {
    headless: boolean;
    channel: RealGpuLaunch["channel"];
    args: string[];
} {
    return {
        headless: LAUNCH_MODES[plan.seat] === "headless",
        channel: plan.channel,
        args: [...plan.args],
    };
}
