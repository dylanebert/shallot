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
// that event fires — calling it again is incoherent).
//
// **Restoring the screen and discharging the exit obligation are two different once-only
// obligations, not one shared flag.** A caller that invokes the returned teardown directly on a
// clean shutdown path and then keeps running is still exposed to a signal arriving *after* that
// direct call — the listener registered below has already suppressed the default termination, so
// if the exit decision were gated on "has the screen already been restored" a later `SIGINT`
// would silently do nothing at all. `didRestore` below guards only the (idempotent) write;
// `didExit` guards the (also idempotent, but tracked independently) exit call, so a signal after
// a direct invocation still finds `didExit` false and still exits. The restore write is also
// wrapped so a throw inside `opts.write` (EPIPE on a closed pty is the realistic case) can never
// suppress the exit call that follows it — the ordering that matters is "restore, then
// unconditionally decide on exit," not "restore, and only decide on exit if restoring didn't
// throw."

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
 * terminating signal (anything other than `"exit"`) also calls `opts.exit` exactly once overall,
 * with the *first* such signal's conventional exit code — in real use `opts.exit` (typically
 * `process.exit`) ends the process synchronously, so a second terminating signal only has a
 * process left to fire against if `opts.exit` didn't actually terminate (a test double, or a
 * caller-supplied non-terminating `exit`), and this module still owes exactly one exit call in
 * that case, not one per signal. That single exit obligation is tracked independently of the
 * (also idempotent) screen restore: a direct invocation of the returned teardown on a clean
 * shutdown path restores the screen but never discharges the exit obligation itself, so a signal
 * arriving afterward still owes exactly one `opts.exit` call — restoring the screen and
 * discharging the exit obligation are two different once-only obligations, not one shared flag
 * (see the module docblock). Returns the same idempotent teardown so a caller can also invoke it
 * directly on a clean shutdown path.
 */
export function installTeardown(opts: TeardownOptions): () => void {
    const events = opts.events ?? DEFAULT_TEARDOWN_SIGNALS;
    let didRestore = false;
    let didExit = false;
    const teardown = (signalEvent?: string) => {
        if (!didRestore) {
            didRestore = true;
            try {
                opts.write(ALT_SCREEN_EXIT);
            } catch {
                // Best-effort: a write failure (EPIPE on a closed pty) must never suppress the
                // exit decision below — restoring the terminal is best-effort, discharging the
                // exit obligation on a terminating signal is not (B1b).
            }
        }
        if (!didExit && signalEvent !== undefined && signalEvent !== "exit") {
            didExit = true;
            opts.exit(SIGNAL_EXIT_CODE[signalEvent] ?? 1);
        }
    };
    for (const event of events) opts.signals.on(event, () => teardown(event));
    return () => teardown();
}
