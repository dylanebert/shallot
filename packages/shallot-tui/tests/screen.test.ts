import { describe, expect, test } from "bun:test";
import { EXIT_ALT_SCREEN, HIDE_CURSOR, SHOW_CURSOR } from "../src/cursor";
import { ALT_SCREEN_ENTER, ALT_SCREEN_EXIT, installTeardown } from "../src/screen";

test("ALT_SCREEN_ENTER hides the cursor before switching buffers, ALT_SCREEN_EXIT restores it after", () => {
    expect(ALT_SCREEN_ENTER.endsWith(HIDE_CURSOR)).toBe(true);
    expect(ALT_SCREEN_EXIT).toBe(SHOW_CURSOR + EXIT_ALT_SCREEN);
});

/** A minimal signal-emitting test double — no real `process.on`, so no real signal is ever touched. */
class FakeSignals {
    private _listeners = new Map<string, (() => void)[]>();
    on(event: string, listener: () => void): this {
        const list = this._listeners.get(event) ?? [];
        list.push(listener);
        this._listeners.set(event, list);
        return this;
    }
    emit(event: string): void {
        for (const l of this._listeners.get(event) ?? []) l();
    }
}

/** Records every call an injected `exit` receives — the proof the process still terminates. */
function fakeExit(): { exit: (code: number) => void; calls: number[] } {
    const calls: number[] = [];
    return { exit: (code: number) => calls.push(code), calls };
}

describe("installTeardown", () => {
    test("writes the restore sequence when a registered signal fires", () => {
        const signals = new FakeSignals();
        const written: string[] = [];
        const { exit } = fakeExit();
        installTeardown({ write: (b) => written.push(b), signals, exit });
        signals.emit("SIGINT");
        expect(written).toEqual([ALT_SCREEN_EXIT]);
    });

    test("a terminating signal also ends the process — registering the listener suppresses Node's own default termination, so this module must restore it", () => {
        const signals = new FakeSignals();
        const written: string[] = [];
        const { exit, calls } = fakeExit();
        installTeardown({ write: (b) => written.push(b), signals, exit });
        signals.emit("SIGINT");
        // the process actually terminates — the bug this repairs was a handler that restored the
        // screen and then just... returned, leaving the process alive on the primary screen with
        // the frame loop still writing into it.
        expect(calls).toEqual([130]);
        signals.emit("SIGTERM");
        // idempotent teardown — a second signal after the first doesn't call exit again either.
        expect(calls).toEqual([130]);
    });

    test("SIGTERM exits with its own conventional code, distinct from SIGINT's", () => {
        const signals = new FakeSignals();
        const { exit, calls } = fakeExit();
        installTeardown({ write: () => {}, signals, exit });
        signals.emit("SIGTERM");
        expect(calls).toEqual([143]);
    });

    test('the "exit" event never calls exit — the process is already tearing down by the time it fires', () => {
        const signals = new FakeSignals();
        const written: string[] = [];
        const { exit, calls } = fakeExit();
        installTeardown({ write: (b) => written.push(b), signals, exit });
        signals.emit("exit");
        expect(written).toEqual([ALT_SCREEN_EXIT]);
        expect(calls).toEqual([]);
    });

    test("is idempotent across multiple signals firing (SIGINT then exit, e.g.)", () => {
        const signals = new FakeSignals();
        const written: string[] = [];
        const { exit, calls } = fakeExit();
        installTeardown({ write: (b) => written.push(b), signals, exit });
        signals.emit("SIGINT");
        signals.emit("exit");
        signals.emit("SIGTERM");
        expect(written).toEqual([ALT_SCREEN_EXIT]);
        expect(calls).toEqual([130]);
    });

    test("the returned teardown can be invoked directly on a clean shutdown, and never calls exit", () => {
        const signals = new FakeSignals();
        const written: string[] = [];
        const { exit, calls } = fakeExit();
        const teardown = installTeardown({ write: (b) => written.push(b), signals, exit });
        teardown();
        expect(written).toEqual([ALT_SCREEN_EXIT]);
        expect(calls).toEqual([]);
        // and a subsequent signal does not double-write or exit
        signals.emit("SIGINT");
        expect(written).toEqual([ALT_SCREEN_EXIT]);
        expect(calls).toEqual([]);
    });

    test("only the caller-supplied events register (no default-signal leakage)", () => {
        const signals = new FakeSignals();
        const written: string[] = [];
        const { exit, calls } = fakeExit();
        installTeardown({ write: (b) => written.push(b), signals, exit, events: ["SIGHUP"] });
        signals.emit("SIGINT");
        expect(written).toEqual([]);
        signals.emit("SIGHUP");
        expect(written).toEqual([ALT_SCREEN_EXIT]);
        // SIGHUP has no entry in the conventional exit-code table — still a real exit (code 1),
        // just without claiming a specific signal's numeric convention.
        expect(calls).toEqual([1]);
    });
});
