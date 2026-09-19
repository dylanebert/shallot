// Placing the display seat's page on the monitor its declaration names, and proving it landed there.
//
// This adds no seat, instrument or declaration file: `SHALLOT_DISPLAY_SEAT` already names one monitor, and
// what it displaces is the assumption that naming it was enough. Hyprland tiles a new window onto whichever
// monitor holds focus, so the page used to present wherever the person happened to be working — DP-1 at
// 239.97 Hz or DP-2 at 143.91 Hz — while the verdict recorded the declared name either way. A 120-frame
// window is then 500 ms or 833 ms, and everything that follows span length rather than allocation moved
// with it. The declaration is made true here instead.
//
// The page is placed by opening it where it belongs rather than by moving it afterwards: the harness gives
// its Chromium a window class of its own, and declares one compositor rule, by a stable name, that sends
// exactly that class to the declared monitor. The rule is replaced rather than accumulated, because a rule
// of the same name supersedes the last one.
//
// A display run takes the declared monitor while it lasts. The page has to be visible there to present at
// that monitor's rate, and the page under test asks for focus and pointer lock as soon as it runs, so it
// takes the keyboard and cursor too. That is the seat, not a defect: these runs are rare and deliberate,
// and presenting on a real display is the measurement.
//
// A rule is a request, so nothing here is trusted: the compositor is read back for where the window
// actually is, and read again after the windows are sampled, because the person may drag it away or switch
// the declared monitor to another workspace mid-run. The geometry comes from the compositor, never
// hard-coded and never from the person's desktop config, so a monitor that is off, renamed or remoded
// refuses by name rather than measuring the wrong display.
//
// Hyprland is the only compositor with a placement here. Another one refuses: an unverified placement is
// exactly the defect this module exists to remove.

import { CAPTURE_CONTRACT } from "./capture";
import { MissingPremise } from "./verdict";

/** one monitor as the compositor reports it. Geometry is layout pixels; `refreshRate` is its current mode. */
export interface SeatMonitor {
    name: string;
    id: number;
    x: number;
    y: number;
    width: number;
    height: number;
    refreshRate: number;
    scale: number;
    /** the workspace this monitor is currently showing; a window elsewhere is not presenting on it. */
    activeWorkspace: number;
}

/** one window as the compositor reports it. */
export interface SeatWindow {
    address: string;
    class: string;
    monitor: number;
    workspace: number;
    floating: boolean;
    mapped: boolean;
    hidden: boolean;
}

/** the window class only this harness's headed Chromium carries, and what the rule matches. */
export const SEAT_WINDOW_CLASS = "shallot-display-seat";

/** the rule's name. One rule, replaced each run, rather than one more rule per run. */
export const SEAT_RULE_NAME = "shallot-display-seat";

/** the page window's size on the person's desktop: the capture contract, so it is shown at its own size. */
const SEAT_WINDOW_SIZE = [CAPTURE_CONTRACT.width, CAPTURE_CONTRACT.height] as const;

/** a declared display ready to be opened on, and the Chromium arguments that land the page there. */
export interface DisplayPlacement {
    monitor: SeatMonitor;
    /** the Chromium arguments the launch must carry for the rule to match. */
    args: readonly string[];
}

/** a page opened on its declared monitor, and the check that it is still there. */
export interface PinnedSeat {
    monitor: SeatMonitor;
    /** the compositor's address for the page's window, the handle the re-check reads. */
    address: string;
    /** Re-read the compositor and refuse if the page is no longer presenting on the declared monitor. */
    stillThere(when: string): Promise<void>;
}

/** Parse `hyprctl monitors -j`. */
export function parseMonitors(json: string): SeatMonitor[] {
    const rows = JSON.parse(json) as Record<string, unknown>[];
    return rows.map((row) => ({
        name: String(row.name),
        id: Number(row.id),
        x: Number(row.x),
        y: Number(row.y),
        width: Number(row.width),
        height: Number(row.height),
        refreshRate: Number(row.refreshRate),
        scale: Number(row.scale),
        activeWorkspace: Number((row.activeWorkspace as { id?: number } | undefined)?.id ?? -1),
    }));
}

/** Parse `hyprctl clients -j`. */
export function parseWindows(json: string): SeatWindow[] {
    const rows = JSON.parse(json) as Record<string, unknown>[];
    return rows.map((row) => ({
        address: String(row.address),
        class: String(row.class ?? ""),
        monitor: Number(row.monitor),
        workspace: Number((row.workspace as { id?: number } | undefined)?.id ?? -1),
        floating: row.floating === true,
        mapped: row.mapped !== false,
        hidden: row.hidden === true,
    }));
}

/**
 * The declared monitor, or the reason the declaration cannot name one. A declaration that names no live
 * monitor is a premise failure and not a measurement: the alternative is measuring some other display and
 * printing this name beside it.
 */
export function declaredMonitor(
    monitors: readonly SeatMonitor[],
    declared: string,
): SeatMonitor | { refused: string } {
    const found = monitors.find((monitor) => monitor.name === declared);
    if (found !== undefined) return found;
    return {
        refused: `SHALLOT_DISPLAY_SEAT declares ${declared}, which is not a live monitor; the compositor reports ${monitors.length === 0 ? "none" : monitors.map((monitor) => monitor.name).join(", ")}`,
    };
}

/**
 * Why the page is not presenting on the declared monitor, or undefined when it is. A window on another
 * monitor, on a workspace that monitor is not showing, unmapped or hidden is not presenting there, and each
 * reads back as its own reason so the refusal names what actually happened.
 */
export function placementRefusal(
    monitor: SeatMonitor,
    window: SeatWindow | undefined,
    when: string,
): string | undefined {
    if (window === undefined)
        return `the page's window is gone from the compositor ${when}, so it cannot be shown to have presented on the declared monitor ${monitor.name}`;
    if (window.monitor !== monitor.id)
        return `the page's window is on monitor ${window.monitor} ${when}, not the declared ${monitor.name} (monitor ${monitor.id}), so the window it presented on is not the one this seat declares`;
    if (window.workspace !== monitor.activeWorkspace)
        return `the page's window is on workspace ${window.workspace} ${when} while ${monitor.name} is showing workspace ${monitor.activeWorkspace}, so the page is not presenting on the declared monitor`;
    if (!window.mapped || window.hidden)
        return `the page's window is ${window.mapped ? "hidden" : "unmapped"} on ${monitor.name} ${when}, so it is not presenting there`;
    return undefined;
}

/**
 * The Lua the compositor runs before the browser launches: one rule, by a stable name, sending this
 * harness's own window class to the declared monitor. `move` is read relative to that monitor, so the
 * window sits in the monitor's bottom-right corner clear of the person's own windows, and
 * `no_initial_focus` keeps the keyboard where it was. The rule is held in a global so it outlives the
 * script that declared it, and a later run of the same name supersedes it.
 */
/**
 * The Lua the compositor runs before the browser launches: one rule, by a stable name, sending this
 * harness's own window class to the declared monitor as a floating window at the capture contract's size,
 * so it is shown at the size it is measured at and does not retile the workspace it lands on. The rule is
 * held in a global so it outlives the script that declared it, and a later run of the same name supersedes
 * it.
 */
export function ruleScript(monitor: SeatMonitor, size: readonly [number, number]): string {
    return [
        "SHALLOT_DISPLAY_SEAT_RULE = hl.window_rule({",
        `    name = ${JSON.stringify(SEAT_RULE_NAME)},`,
        `    match = { class = ${JSON.stringify(`^(${SEAT_WINDOW_CLASS})$`)} },`,
        `    monitor = ${JSON.stringify(monitor.name)},`,
        "    float = true,",
        `    size = ${JSON.stringify(`${size[0]} ${size[1]}`)},`,
        "})",
        "return 1",
    ].join("\n");
}

async function hyprctl(args: readonly string[]): Promise<string> {
    const child = Bun.spawn(["hyprctl", ...args], { stdout: "pipe", stderr: "pipe" });
    const [out, err] = await Promise.all([
        new Response(child.stdout).text(),
        new Response(child.stderr).text(),
    ]);
    await child.exited;
    if (child.exitCode !== 0 || out.startsWith("error:"))
        throw new MissingPremise(
            `the harness cannot read the compositor: hyprctl ${args[0]} exited ${child.exitCode} with ${(out + err).trim().split("\n")[0] || "no output"}`,
        );
    return out;
}

const monitors = async () => parseMonitors(await hyprctl(["monitors", "-j"]));
const seatWindows = async () =>
    (await hyprctl(["clients", "-j"]).then(parseWindows)).filter(
        (row) => row.class === SEAT_WINDOW_CLASS,
    );

/** how long the window is given to map before the compositor is read back. */
const MAP_TIMEOUT_MS = 5_000;

/**
 * Declare the placement for the monitor `declared` names, and return the arguments that make a launch land
 * there. Called before the browser launches, because a window that opens where it belongs never crosses the
 * person's workspace at all.
 */
export async function openOnDisplay(declared: string): Promise<DisplayPlacement> {
    if (process.platform !== "linux" || !process.env.HYPRLAND_INSTANCE_SIGNATURE)
        throw new MissingPremise(
            `the display seat declares ${declared}, but this host has no compositor placement: only Hyprland can be asked to open the page on a named monitor here, and a seat that cannot be verified is the defect this row refuses`,
        );
    const monitor = declaredMonitor(await monitors(), declared);
    if ("refused" in monitor) throw new MissingPremise(monitor.refused);
    const left = await seatWindows();
    if (left.length > 0)
        throw new MissingPremise(
            `${left.length} window of this harness's own class ${SEAT_WINDOW_CLASS} is already open, so the page's window could not be told apart from it; close it and run again`,
        );
    await hyprctl(["eval", ruleScript(monitor, SEAT_WINDOW_SIZE)]);
    return { monitor, args: [`--class=${SEAT_WINDOW_CLASS}`] };
}

/**
 * Read back where the page's window actually is, and refuse unless it is presenting on the declared
 * monitor. A rule is a request; this is the evidence.
 */
export async function confirmOnDisplay(placement: DisplayPlacement): Promise<PinnedSeat> {
    const { monitor } = placement;
    const deadline = Date.now() + MAP_TIMEOUT_MS;
    let open = await seatWindows();
    while (open.length === 0 && Date.now() < deadline) {
        await Bun.sleep(100);
        open = await seatWindows();
    }
    if (open.length !== 1)
        throw new MissingPremise(
            `the compositor reports ${open.length} windows of the page's class ${SEAT_WINDOW_CLASS}, not one, so which window presented on ${monitor.name} cannot be established`,
        );
    const refusal = placementRefusal(monitor, open[0], "when it opened");
    if (refusal !== undefined) throw new MissingPremise(refusal);
    const address = open[0].address;
    return {
        monitor,
        address,
        stillThere: async (when: string) => {
            // The monitor is re-read too, not just the window: the person is using this desktop, and a
            // workspace switch on the declared monitor hides the page as surely as dragging it away does.
            const current = declaredMonitor(await monitors(), monitor.name);
            if ("refused" in current) throw new MissingPremise(current.refused);
            const still = (await seatWindows()).find((row) => row.address === address);
            const moved = placementRefusal(current, still, when);
            if (moved !== undefined) throw new MissingPremise(moved);
        },
    };
}
