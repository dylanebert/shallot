// Alt-screen enter/exit with cursor restore on signal. The failure mode this guards is a process
// that dies mid-frame (Ctrl-C, a kill, an uncaught throw) leaving the user's real terminal on the
// alternate buffer with its cursor hidden — every teardown path writes the same restore sequence,
// so the terminal is never stuck.

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

export interface TeardownOptions {
    /** Writes bytes to the real terminal — typically `process.stdout.write`. */
    readonly write: (bytes: string) => void;
    /** The signal-emitting object to listen on — typically `process`, injected for testability. */
    readonly signals: SignalSource;
    /** Overrides `DEFAULT_TEARDOWN_SIGNALS`. */
    readonly events?: readonly string[];
}

/**
 * Registers a teardown listener on every event in `events` (default `DEFAULT_TEARDOWN_SIGNALS`)
 * that writes `ALT_SCREEN_EXIT` exactly once, however many of those events fire — `SIGINT` and
 * `exit` both fire on a Ctrl-C under Node, and a double-write is harmless but a signal handler
 * that assumes it fires once is not. Returns the same idempotent teardown so a caller can also
 * invoke it directly on a clean shutdown path.
 */
export function installTeardown(opts: TeardownOptions): () => void {
    const events = opts.events ?? DEFAULT_TEARDOWN_SIGNALS;
    let didTeardown = false;
    const teardown = () => {
        if (didTeardown) return;
        didTeardown = true;
        opts.write(ALT_SCREEN_EXIT);
    };
    for (const event of events) opts.signals.on(event, teardown);
    return teardown;
}
