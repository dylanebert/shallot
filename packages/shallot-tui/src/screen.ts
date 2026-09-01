// Alt-screen enter/exit with cursor restore on signal. The failure mode this guards is a process
// that dies mid-frame (Ctrl-C, a kill, an uncaught throw) leaving the user's real terminal on the
// alternate buffer with its cursor hidden — every teardown path writes the same restore sequence,
// so the terminal is never stuck.
//
// Registering a listener for `SIGINT`/`SIGTERM` removes Node/Bun's *default* disposition for that
// signal (terminate the process) — from that point the process only exits if something in it
// explicitly makes it exit. A handler that writes the restore bytes and returns without exiting
// therefore leaves the process alive on the primary screen with the frame loop still running,
// writing escape bytes into what is now the user's live shell: worse than the failure this module
// exists to prevent. So `installTeardown` doesn't just restore the screen on a terminating signal
// — it also owns telling the caller-supplied `exit` to actually end the process. The caller
// supplies `exit` (typically `process.exit`) rather than this module calling `process.exit`
// itself, because only the caller knows whether this process's lifetime is its own to end: a
// caller embedding this inside a larger process must not have a terminal helper terminate it
// uninvited. `exit` is never called for `"exit"` (the process is already tearing down by the time
// that event fires — calling it again is incoherent) nor for a direct invocation of the returned
// teardown on a clean shutdown path (the caller invoking it directly is already handling its own
// exit on its own terms).

import { ENTER_ALT_SCREEN, EXIT_ALT_SCREEN, HIDE_CURSOR, SHOW_CURSOR } from "./cursor";

/** Bytes that enter the alternate screen and hide the cursor — write once, at session start. */
export const ALT_SCREEN_ENTER = ENTER_ALT_SCREEN + HIDE_CURSOR;

/** Bytes that restore the cursor and leave the alternate screen — the teardown sequence. */
export const ALT_SCREEN_EXIT = SHOW_CURSOR + EXIT_ALT_SCREEN;

/** The subset of Node's `process` this module needs: registering a listener by event name. */
export interface SignalSource {
    on(event: string, listener: () => void): unknown;
}

/** Signals a teardown must run on to guarantee the terminal isn't left on the alt screen. */
export const DEFAULT_TEARDOWN_SIGNALS: readonly string[] = ["SIGINT", "SIGTERM", "exit"];

/**
 * Conventional shell exit code (128 + signal number) for every terminating signal this module
 * registers by default. An event outside this table (a caller-supplied custom event, e.g.
 * `SIGHUP`) still exits — via the `?? 1` fallback below — just without claiming a specific
 * signal's numeric convention.
 */
const SIGNAL_EXIT_CODE: Readonly<Record<string, number>> = { SIGINT: 130, SIGTERM: 143 };

export interface TeardownOptions {
    /** Writes bytes to the real terminal — typically `process.stdout.write`. */
    readonly write: (bytes: string) => void;
    /** The signal-emitting object to listen on — typically `process`, injected for testability. */
    readonly signals: SignalSource;
    /**
     * Ends the process after the restore bytes are written for a terminating signal — typically
     * `process.exit`, injected because only the caller knows whether this call owns the
     * process's lifetime. Required: a teardown that restores the screen but never exits leaves
     * Ctrl-C's default termination suppressed with nothing terminating in its place, which is the
     * bug this module exists to not reintroduce.
     */
    readonly exit: (code: number) => void;
    /** Overrides `DEFAULT_TEARDOWN_SIGNALS`. */
    readonly events?: readonly string[];
}

/**
 * Registers a teardown listener on every event in `events` (default `DEFAULT_TEARDOWN_SIGNALS`)
 * that writes `ALT_SCREEN_EXIT` exactly once, however many of those events fire — a double-write
 * is harmless but a signal handler that assumes it fires once is not. A listener firing from a
 * terminating signal (anything other than `"exit"`) also calls `opts.exit` once, with that
 * signal's conventional exit code, restoring the termination that registering the listener
 * removed. Returns the same idempotent teardown so a caller can also invoke it directly on a
 * clean shutdown path — a direct invocation never calls `exit`, since the caller invoking it is
 * already mid-shutdown on its own terms.
 */
export function installTeardown(opts: TeardownOptions): () => void {
    const events = opts.events ?? DEFAULT_TEARDOWN_SIGNALS;
    let didTeardown = false;
    const teardown = (signalEvent?: string) => {
        if (didTeardown) return;
        didTeardown = true;
        opts.write(ALT_SCREEN_EXIT);
        if (signalEvent !== undefined && signalEvent !== "exit") {
            opts.exit(SIGNAL_EXIT_CODE[signalEvent] ?? 1);
        }
    };
    for (const event of events) opts.signals.on(event, () => teardown(event));
    return () => teardown();
}
