import { UnsupportedError } from "./gpu";

/** Refuse Windows hosts that Dawn cannot load before its native addon is imported. */
function assertDawnHost(platform: string, execPath: string): void {
    const executable = execPath.split(/[\\/]/).at(-1)?.toLowerCase();
    if (platform === "win32" && executable !== "node.exe") {
        throw new UnsupportedError(
            "On Windows, dawn.node imports N-API directly from node.exe (run 38151144247); " +
                "use Node, or Bun run as a copy named node.exe.",
        );
    }
}

/** Check the native host before asking the caller to import Dawn. */
export async function loadDawn<T>(
    platform: string,
    execPath: string,
    load: () => Promise<T>,
): Promise<T> {
    assertDawnHost(platform, execPath);
    return load();
}
