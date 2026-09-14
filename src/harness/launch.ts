// Host launch declarations, kept separate from seat policy. A host declares only whether it has a launch
// path and whether a real adapter has actually been proven there; the launch mode is policy and lives
// here in code, so no declaration can ask for a headed browser or change what a seat means.

import type { RealGpuLaunch } from "./browser";
import launchFloor from "./browser.json" with { type: "json" };
import hosts from "./launch.json" with { type: "json" };

/** the one launch mode the harness has. There is no headed fallback: a host that cannot reach a real
 *  adapter headlessly refuses, rather than opening a window to pass. */
export const LAUNCH_MODE = "headless" as const;

/** whether a real adapter has actually been observed on a host, or is only declared. `unproven` never
 *  grants a seat — the driver still has to observe a real adapter at run time. */
export type AdapterEvidence = "proven" | "unproven";

/** one host's launch declaration: pure data, with no launch mode or capability of its own. */
export interface HostLaunchDeclaration {
    adapterEvidence: AdapterEvidence;
    note: string;
}

/** the declared hosts, keyed by `process.platform`. An absent host has no launch path and refuses. */
export const HOST_LAUNCHES: Readonly<Record<string, HostLaunchDeclaration>> = hosts as Readonly<
    Record<string, HostLaunchDeclaration>
>;

/** the real-GPU launch floor every host shares. */
export const LAUNCH_FLOOR: RealGpuLaunch = launchFloor as RealGpuLaunch;

/** a resolved launch: the fixed mode plus the shared floor, for one declared host. */
export interface LaunchPlan {
    host: string;
    mode: typeof LAUNCH_MODE;
    channel: RealGpuLaunch["channel"];
    args: readonly string[];
    adapterEvidence: AdapterEvidence;
}

/** Resolve a host's launch plan, or refuse a host with no declaration. */
export function launchPlan(host: string): LaunchPlan | { refused: string } {
    const declaration = HOST_LAUNCHES[host];
    if (declaration === undefined) {
        return {
            refused: `no declared headless Chromium launch path for host ${host}; declared hosts are ${Object.keys(HOST_LAUNCHES).sort().join(", ")}`,
        };
    }
    return {
        host,
        mode: LAUNCH_MODE,
        channel: LAUNCH_FLOOR.channel,
        args: LAUNCH_FLOOR.args,
        adapterEvidence: declaration.adapterEvidence,
    };
}

/** the Playwright launch options for a plan. `headless` comes from policy, never from host data. */
export function launchOptions(plan: LaunchPlan): {
    headless: true;
    channel: RealGpuLaunch["channel"];
    args: string[];
} {
    return { headless: true, channel: plan.channel, args: [...plan.args] };
}
