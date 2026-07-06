declare const Bun: {
    file: (path: string) => {
        text: () => Promise<string>;
        arrayBuffer: () => Promise<ArrayBuffer>;
    };
};

const isBun = typeof Bun !== "undefined";

/** the execution environment: `"web"` in a browser, `"headless"` under Bun (tests, tooling). */
export type Runtime = "web" | "headless";
/** the execution environment: `"web"` in a browser, `"headless"` under Bun (tests, tooling). */
export const Runtime: Runtime = isBun ? "headless" : "web";

/** monotonic high-resolution timestamp in milliseconds (`performance.now`). */
export const now = (): number => performance.now();

/**
 * schedule `callback` for the next frame — `requestAnimationFrame` on web, a `setTimeout(0)` when
 * headless. The web path forwards rAF's frame-start timestamp; the headless path calls with no argument,
 * so consumers treat it as the sim timebase either way.
 */
export const requestFrame: (callback: (timestamp?: number) => void) => void = isBun
    ? (cb) => setTimeout(cb, 0)
    : (cb) => requestAnimationFrame(cb);

/** read a text file — `Bun.file` when headless, `fetch` on web (throws on a non-ok response). */
export async function readFile(path: string): Promise<string> {
    if (isBun) return Bun.file(path).text();
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
    return response.text();
}

/** read a binary file as an `ArrayBuffer` — `Bun.file` when headless, `fetch` on web (throws on a non-ok response). */
export async function readBinary(path: string): Promise<ArrayBuffer> {
    if (isBun) return Bun.file(path).arrayBuffer();
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
    return response.arrayBuffer();
}
