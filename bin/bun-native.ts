/** The bridge surface Shallot uses; typed here so the peer's source never joins this project's typecheck. */
export interface Native {
    setupGlobals(options?: { libPath?: string }): Promise<void>;
    globals(): void;
    createGPUInstance(libPath?: string): GPU & { destroy(): undefined };
    globalConstructors: Record<string, unknown>;
}

const PEER = "bun-webgpu";

/** Load the optional bun-webgpu peer, the Dawn bridge for native GPU setup under Bun. */
export async function loadNative(): Promise<Native> {
    if (typeof Bun === "undefined") throw new Error("Shallot native setup requires Bun");
    return import(PEER);
}
