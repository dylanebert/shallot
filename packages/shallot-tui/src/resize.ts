// Terminal size and resize handling. Pure functions over an injected stream-shaped object, never
// a direct `process.stdout` read — a caller (the bin layer, out of this unit's scope) supplies the
// real stream; tests supply a plain object, so this is testable with no real TTY on any seat.

/** The subset of a Node writable stream this module reads: current terminal dimensions. */
export interface SizeSource {
    readonly columns?: number;
    readonly rows?: number;
}

/** The subset of a Node writable stream this module subscribes to: the `"resize"` event. */
export interface ResizeSource extends SizeSource {
    on(event: "resize", listener: () => void): unknown;
    off?(event: "resize", listener: () => void): unknown;
}

export interface TerminalSize {
    readonly width: number;
    readonly height: number;
}

const DEFAULT_SIZE: TerminalSize = { width: 80, height: 24 };

/**
 * Reads `stream`'s current size, falling back to 80x24 when the stream reports no usable
 * dimensions (a non-tty stream, or `columns`/`rows` of 0 or `undefined`) — the same floor the
 * portability ladder's `glyph` tier targets at its cheapest.
 */
export function terminalSize(
    stream: SizeSource,
    fallback: TerminalSize = DEFAULT_SIZE,
): TerminalSize {
    const width = stream.columns && stream.columns > 0 ? stream.columns : fallback.width;
    const height = stream.rows && stream.rows > 0 ? stream.rows : fallback.height;
    return { width, height };
}

/**
 * Subscribes `callback` to `stream`'s resize events, calling it with the new size each time.
 * Returns an unsubscribe function. `stream.off` is optional (mirrors Node's `EventEmitter`, which
 * always has it, while a minimal test double may not) — when absent, the listener is left
 * attached and the returned unsubscribe is a no-op, which is only reachable from a caller that
 * built its own non-conforming stream.
 */
export function onResize(stream: ResizeSource, callback: (size: TerminalSize) => void): () => void {
    const listener = () => callback(terminalSize(stream));
    stream.on("resize", listener);
    return () => stream.off?.("resize", listener);
}
