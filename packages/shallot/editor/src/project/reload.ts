import type { Plugin } from "@dylanebert/shallot";
import type { Manifest } from "./manifest";

/** the project's re-imported local plugins after a code hot reload, plus the (unchanged) manifest */
export interface ProjectReload {
    locals: { name: string; plugin: Plugin }[];
    manifest: Manifest;
}

let _handler: ((next: ProjectReload) => void) | null = null;

/**
 * subscribe to project-code hot reloads. The editor host registers one handler that swaps the
 * reloaded plugins onto the live State (falling back to a rebuild on a shape change). The generated
 * `virtual:project` module's HMR accept is the sole emitter — see {@link emitProjectReload}.
 */
export function onProjectReload(fn: (next: ProjectReload) => void): void {
    _handler = fn;
}

/** the `virtual:project` HMR accept calls this with the freshly re-discovered project plugins. */
export function emitProjectReload(next: ProjectReload): void {
    _handler?.(next);
}
