/// <reference types="svelte" />
/// <reference types="vite/client" />

declare module "virtual:project" {
    import type { Plugin } from "@dylanebert/shallot";
    import type { Manifest } from "./project/manifest";

    const project: {
        dir: string | null;
        scene: string | null;
        capacity: number | null;
        scenes: string[];
        manifest: Manifest;
        locals: { name: string; plugin: Plugin }[];
        plugins: Plugin[];
    };
    export default project;
}

declare module "*.scene?raw" {
    const content: string;
    export default content;
}
