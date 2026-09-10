// Resolves the Box3D C reference the physics generators build against. The pin lives in
// crates/physics/reference.json; the checkout is cloned on demand into the user cache at exactly that
// commit, so no generator reaches outside this repo for it. Offline with no cached checkout, it refuses
// with the command that would fetch it.
//
// Usage: bun run crates/physics/scripts/reference.ts --where   (prints the resolved path, clones nothing)

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

interface Pin {
    url: string;
    commit: string;
    branch: string;
}

const PIN_FILE = resolve(import.meta.dir, "../reference.json");

export function pin(): Pin {
    return JSON.parse(readFileSync(PIN_FILE, "utf8"));
}

/** the cache path of the pinned checkout, whether or not it exists yet. */
export function referenceDir(p: Pin = pin()): string {
    const cache = process.env.XDG_CACHE_HOME || join(homedir(), ".cache");
    return join(cache, "shallot", "reference", `box3d-${p.commit}`);
}

function git(args: string[], cwd?: string): boolean {
    return spawnSync("git", args, { cwd, stdio: "inherit" }).status === 0;
}

/** the pinned checkout's path, cloning it at the pinned commit first when absent. Exits with the
 *  remedy when the clone fails (offline, or the commit gone from the remote). */
export function ensureReference(p: Pin = pin()): string {
    const dir = referenceDir(p);
    if (existsSync(join(dir, ".git"))) return dir;
    const tmp = `${dir}.${process.pid}.tmp`;
    rmSync(tmp, { recursive: true, force: true });
    mkdirSync(dirname(dir), { recursive: true });
    console.log(`[physics/reference] cloning ${p.url} @ ${p.commit} into ${dir}`);
    const ok =
        git(["clone", "--branch", p.branch, "--single-branch", p.url, tmp]) &&
        git(["-c", "advice.detachedHead=false", "checkout", p.commit], tmp);
    if (!ok) {
        rmSync(tmp, { recursive: true, force: true });
        console.error(`box3d reference unavailable: could not clone ${p.url} @ ${p.commit}.`);
        console.error(
            `Remedy: connect to the network and rerun, or clone it by hand:\n  git clone --branch ${p.branch} ${p.url} ${dir} && git -C ${dir} checkout ${p.commit}`,
        );
        process.exit(1);
    }
    renameSync(tmp, dir);
    return dir;
}

if (import.meta.main) {
    if (process.argv[2] !== "--where") {
        console.error("usage: bun run crates/physics/scripts/reference.ts --where");
        process.exit(1);
    }
    const dir = referenceDir();
    console.log(`${dir} (${existsSync(join(dir, ".git")) ? "exists" : "absent"})`);
}
