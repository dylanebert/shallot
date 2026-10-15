declare const Bun: {
    file: (path: string) => {
        text: () => Promise<string>;
        arrayBuffer: () => Promise<ArrayBuffer>;
    };
};

const isBun = typeof Bun !== "undefined";

export type Runtime = "web" | "headless";
/** the execution environment: `"web"` in a browser, `"headless"` under Bun (tests, tooling). */
export const Runtime: Runtime = isBun ? "headless" : "web";

/** monotonic high-resolution timestamp in milliseconds (`performance.now`). */
export const now = (): number => performance.now();

/**
 * Schedule with rAF's millisecond frame-start timestamp on web, or `setTimeout(0)` without a
 * timestamp under Bun. Callers needing a headless clock must sample it themselves.
 */
export const requestFrame: (callback: (timestamp?: number) => void) => void = isBun
    ? (cb) => setTimeout(cb, 0)
    : (cb) => requestAnimationFrame(cb);

/** read a text file: `Bun.file` when headless, `fetch` on web (throws on a non-ok response). */
export async function readFile(path: string): Promise<string> {
    if (isBun) return Bun.file(path).text();
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
    return response.text();
}

/** read a binary file as an `ArrayBuffer`: `Bun.file` when headless, `fetch` on web (throws on a non-ok response). */
export async function readBinary(path: string): Promise<ArrayBuffer> {
    if (isBun) return Bun.file(path).arrayBuffer();
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
    return response.arrayBuffer();
}
