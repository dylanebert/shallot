// Host launch declarations, kept separate from seat policy. A host declares only evidence: whether a real
// adapter has actually been observed in a headed launch there. The display seat is always headed, because
// presenting on a monitor is its premise.

import floor from "./browser.json" with { type: "json" };
import hosts from "./launch.json" with { type: "json" };

/** whether a real adapter has actually been observed on a host, or is only declared. `unproven` never
 *  grants a seat — the run still has to observe a real adapter. */
export type AdapterEvidence = "proven" | "unproven";

/** one host's launch declaration: pure data, with no launch mode or capability of its own. */
export interface HostLaunchDeclaration {
    adapterEvidence: AdapterEvidence;
    note: string;
}

/** the declared hosts, keyed by `process.platform`. An absent host has no launch path and refuses. */
const HOST_LAUNCHES: Readonly<Record<string, HostLaunchDeclaration>> = hosts as Readonly<
    Record<string, HostLaunchDeclaration>
>;

/** a resolved headed launch of the display seat on one declared host, with the real-GPU floor. */
export interface LaunchPlan {
    host: string;
    seat: "display";
    channel: "chromium";
    args: readonly string[];
    adapterEvidence: AdapterEvidence;
}

/** Resolve a host's display launch plan, or refuse a host with no declaration. */
export function launchPlan(host: string): LaunchPlan | { refused: string } {
    const declaration = HOST_LAUNCHES[host];
    if (declaration === undefined) {
        return {
            refused: `no declared headed Chromium launch path for host ${host}; declared hosts are ${Object.keys(HOST_LAUNCHES).sort().join(", ")}`,
        };
    }
    return {
        host,
        seat: "display",
        channel: floor.channel as "chromium",
        args: floor.args,
        adapterEvidence: declaration.adapterEvidence,
    };
}
