import { UnsupportedError } from "./gpu";

/** Refuse Windows hosts that Dawn cannot load before its native addon is imported. */
function assertDawnHost(platform: string, execPath: string): void {
    const executable = execPath.split(/[\\/]/).at(-1)?.toLowerCase();
    if (platform === "win32" && executable !== "node.exe") {
        throw new UnsupportedError(
            "On Windows, dawn.node imports N-API directly from node.exe, so a host with another name crashes loading it; " +
                "use Node, or run Bun from a copy named node.exe.",
        );
    }
}

/** Resolve the optional peer itself before loading Dawn, so internal module failures keep their cause. */
export async function loadDawn<T>(
    platform: string,
    execPath: string,
    load: () => Promise<T>,
    resolvePeer: () => string = () => import.meta.resolve("webgpu"),
): Promise<T> {
    assertDawnHost(platform, execPath);
    try {
        resolvePeer();
    } catch (cause) {
        const code = (cause as { code?: unknown } | null)?.code;
        if (code === "MODULE_NOT_FOUND" || code === "ERR_MODULE_NOT_FOUND") {
            throw new UnsupportedError(
                'The optional "webgpu" peer is missing; install it with `bun add webgpu` before calling setupGlobals().',
                [],
                { cause },
            );
        }
        throw cause;
    }
    return load();
}
