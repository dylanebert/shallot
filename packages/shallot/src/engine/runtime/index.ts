declare const Bun: {
    file: (path: string) => {
        text: () => Promise<string>;
        arrayBuffer: () => Promise<ArrayBuffer>;
    };
};

const isBun = typeof Bun !== "undefined";

export type Runtime = "web" | "headless";
export const Runtime: Runtime = isBun ? "headless" : "web";

export const now = (): number => performance.now();

export const requestFrame: (callback: () => void) => void = isBun
    ? (cb) => setTimeout(cb, 0)
    : (cb) => requestAnimationFrame(cb);

export async function readFile(path: string): Promise<string> {
    if (isBun) return Bun.file(path).text();
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
    return response.text();
}

export async function readBinary(path: string): Promise<ArrayBuffer> {
    if (isBun) return Bun.file(path).arrayBuffer();
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
    return response.arrayBuffer();
}
