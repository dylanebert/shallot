import { resolve } from "node:path";
import { runVite } from "./vite-command";

/** Build through the project's own Vite config. */
export async function buildWeb(
    projectDir: string,
    output: { outDir?: string; sourcemap?: boolean } = {},
    args: string[] = [],
): Promise<void> {
    const outputArgs = [
        ...(output.outDir === undefined ? [] : ["--outDir", resolve(projectDir, output.outDir)]),
        ...(output.sourcemap === undefined ? [] : [`--sourcemap=${output.sourcemap}`]),
    ];
    const status = runVite(projectDir, "build", [...args, ...outputArgs]);
    if (status !== 0) throw new Error(`vite build exited with status ${status}`);
}
