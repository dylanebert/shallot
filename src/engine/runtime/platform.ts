declare const Bun: {
    file: (path: string) => {
        text: () => Promise<string>;
        arrayBuffer: () => Promise<ArrayBuffer>;
    };
};

const isBun = typeof Bun !== "undefined";

export type Runtime = "web" | "headless";
/** the execution environment: `"headless"` under Bun (tests, tooling), otherwise `"web"`, including under Node. */
export const Runtime: Runtime = isBun ? "headless" : "web";

/** monotonic high-resolution timestamp in milliseconds (`performance.now`). */
export const now = (): number => performance.now();

/**
 * Schedule with `setTimeout(0)` and no timestamp under Bun; otherwise with `requestAnimationFrame` and
 * its millisecond frame-start timestamp, which Node has only where its host defines one. Callers
 * needing a headless clock must sample it themselves.
 */
export const requestFrame: (callback: (timestamp?: number) => void) => void = isBun
    ? (cb) => setTimeout(cb, 0)
    : (cb) => requestAnimationFrame(cb);

/** read a text file: `Bun.file` under Bun, otherwise `fetch` (throws on a non-ok response). */
export async function readFile(path: string): Promise<string> {
    if (isBun) return Bun.file(path).text();
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
    return response.text();
}

/** read a binary file as an `ArrayBuffer`: `Bun.file` under Bun, otherwise `fetch` (throws on a non-ok response). */
export async function readBinary(path: string): Promise<ArrayBuffer> {
    if (isBun) return Bun.file(path).arrayBuffer();
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
    return response.arrayBuffer();
}
