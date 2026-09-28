import type { Plugin } from "vite";
/**
 * The Vite plugin set a Shallot project needs: `plugins: [shallot()]`. TypeGPU stays a separate
 * plugin entry alongside project support, so its hooks are composed rather than overwritten. Its
 * transform must reach engine source inside `node_modules`; a second pass corrupts its metadata.
 */
export declare function shallot(projectDir?: string): Plugin[];
