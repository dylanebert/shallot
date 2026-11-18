import { resolve } from "node:path";
import { bunChild } from "./bun-child";
import { spawnPackedInstall } from "./packed-install-spawn";

export function packCheckout(args: string[]): string {
    const root = resolve(import.meta.dir, "..");
    spawnPackedInstall(bunChild(["run", "prepare"]), root, "building packed tooling");
    return spawnPackedInstall(
        bunChild(["pm", "pack", "--ignore-scripts", ...args]),
        root,
        "packing checkout",
    ).trim();
}

if (import.meta.main) process.stdout.write(packCheckout(process.argv.slice(2)));
