// Host launch declarations, kept separate from seat policy. A host declares only evidence: whether a real
// adapter has actually been observed there in each launch mode. The launch mode is derived here from that
// evidence and the seat, never declared a second time, so no declaration can name a mode its evidence does
// not support or change what a seat means.

import type { RealGpuLaunch } from "./browser";
import launchFloor from "./browser.json" with { type: "json" };
import hosts from "./launch.json" with { type: "json" };

/** the browser seats. */
export const LAUNCH_SEATS = ["chromium", "display"] as const;
export type LaunchSeat = (typeof LAUNCH_SEATS)[number];
export type LaunchMode = "headless" | "headed";

/** the window class a headed `chromium` launch carries. The seat's compositor holds a standing rule for it,
 *  a silent special workspace with no focus, and the driver reads back that the rule took. */
export const HIDDEN_WINDOW_CLASS = "kex-gate";

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

/** a resolved launch: one seat plus the shared floor, for one declared host. Its mode is always
 *  {@link launchMode} of its seat and host, never a field of its own, so no plan can disagree with them. */
export interface LaunchPlan {
    host: string;
    seat: LaunchSeat;
    channel: RealGpuLaunch["channel"];
    args: readonly string[];
    /** the host's evidence in this plan's mode. */
    adapterEvidence: AdapterEvidence;
}

/**
 * The mode a seat launches in on a host. `display` is always headed: presenting on a monitor is its
 * premise. `chromium` takes nothing from the desktop, so it is headless wherever headless is proven, and
 * headed and hidden only where headed alone is proven; with neither proven it stays headless and the
 * observed adapter decides.
 */
export function launchMode(host: string, seat: LaunchSeat): LaunchMode {
    if (seat === "display") return "headed";
    const evidence = HOST_LAUNCHES[host]?.adapterEvidence;
    return evidence?.headless !== "proven" && evidence?.headed === "proven" ? "headed" : "headless";
}

/** Resolve a host's launch plan for a browser seat, or refuse a host with no declaration. */
export function launchPlan(
    host: string,
    seat: LaunchSeat = "chromium",
): LaunchPlan | { refused: string } {
    const mode = launchMode(host, seat);
    const declaration = HOST_LAUNCHES[host];
    if (declaration === undefined) {
        return {
            refused: `no declared ${mode} Chromium launch path for host ${host}; declared hosts are ${Object.keys(HOST_LAUNCHES).sort().join(", ")}`,
        };
    }
    return {
        host,
        seat,
        channel: LAUNCH_FLOOR.channel,
        args: LAUNCH_FLOOR.args,
        adapterEvidence: declaration.adapterEvidence[mode],
    };
}

/** the Playwright launch options for a plan. `headless` is derived from the plan's seat and host, never
 *  from a caller, and a headed `chromium` launch always carries {@link HIDDEN_WINDOW_CLASS}. */
export function launchOptions(plan: LaunchPlan): {
    headless: boolean;
    channel: RealGpuLaunch["channel"];
    args: string[];
} {
    const mode = launchMode(plan.host, plan.seat);
    return {
        headless: mode === "headless",
        channel: plan.channel,
        args:
            plan.seat === "chromium" && mode === "headed"
                ? [...plan.args, `--class=${HIDDEN_WINDOW_CLASS}`]
                : [...plan.args],
    };
}
