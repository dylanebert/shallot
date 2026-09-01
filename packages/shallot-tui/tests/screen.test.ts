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

describe("installTeardown", () => {
    test("writes the restore sequence when a registered signal fires", () => {
        const signals = new FakeSignals();
        const written: string[] = [];
        installTeardown({ write: (b) => written.push(b), signals });
        signals.emit("SIGINT");
        expect(written).toEqual([ALT_SCREEN_EXIT]);
    });

    test("is idempotent across multiple signals firing (SIGINT then exit, e.g.)", () => {
        const signals = new FakeSignals();
        const written: string[] = [];
        installTeardown({ write: (b) => written.push(b), signals });
        signals.emit("SIGINT");
        signals.emit("exit");
        signals.emit("SIGTERM");
        expect(written).toEqual([ALT_SCREEN_EXIT]);
    });

    test("the returned teardown can be invoked directly on a clean shutdown", () => {
        const signals = new FakeSignals();
        const written: string[] = [];
        const teardown = installTeardown({ write: (b) => written.push(b), signals });
        teardown();
        expect(written).toEqual([ALT_SCREEN_EXIT]);
        // and a subsequent signal does not double-write
        signals.emit("SIGINT");
        expect(written).toEqual([ALT_SCREEN_EXIT]);
    });

    test("only the caller-supplied events register (no default-signal leakage)", () => {
        const signals = new FakeSignals();
        const written: string[] = [];
        installTeardown({ write: (b) => written.push(b), signals, events: ["SIGHUP"] });
        signals.emit("SIGINT");
        expect(written).toEqual([]);
        signals.emit("SIGHUP");
        expect(written).toEqual([ALT_SCREEN_EXIT]);
    });
});
