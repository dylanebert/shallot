import { describe, expect, test } from "bun:test";
import { onResize, terminalSize } from "./resize";

describe("terminalSize", () => {
    test("reads columns/rows straight off the stream", () => {
        expect(terminalSize({ columns: 100, rows: 40 })).toEqual({ width: 100, height: 40 });
    });

    test("falls back to 80x24 when columns/rows are absent", () => {
        expect(terminalSize({})).toEqual({ width: 80, height: 24 });
    });

    test("falls back when columns/rows are zero (a non-tty stream's typical reading)", () => {
        expect(terminalSize({ columns: 0, rows: 0 })).toEqual({ width: 80, height: 24 });
    });

    test("honors a caller-supplied fallback", () => {
        expect(terminalSize({}, { width: 40, height: 12 })).toEqual({ width: 40, height: 12 });
    });
});

/** A minimal EventEmitter-shaped test double — no real TTY, no `process` touched. */
class FakeStream {
    columns = 80;
    rows = 24;
    private _listeners: (() => void)[] = [];
    on(event: "resize", listener: () => void): this {
        if (event === "resize") this._listeners.push(listener);
        return this;
    }
    off(event: "resize", listener: () => void): this {
        if (event === "resize") this._listeners = this._listeners.filter((l) => l !== listener);
        return this;
    }
    emitResize(columns: number, rows: number): void {
        this.columns = columns;
        this.rows = rows;
        for (const l of this._listeners) l();
    }
}

describe("onResize", () => {
    test("calls back with the new size when the stream emits resize", () => {
        const stream = new FakeStream();
        const seen: { width: number; height: number }[] = [];
        onResize(stream, (size) => seen.push(size));
        stream.emitResize(120, 50);
        expect(seen).toEqual([{ width: 120, height: 50 }]);
    });

    test("the returned unsubscribe stops further callbacks", () => {
        const stream = new FakeStream();
        const seen: { width: number; height: number }[] = [];
        const unsubscribe = onResize(stream, (size) => seen.push(size));
        unsubscribe();
        stream.emitResize(120, 50);
        expect(seen).toEqual([]);
    });
});
