/// <reference types="svelte" />
/// <reference types="vite/client" />

declare module "virtual:project" {
    import type { DiscoveredPlugin } from "./project";
    import type { Config } from "@dylanebert/shallot";
    const project: {
        custom: DiscoveredPlugin[];
        scenes: string[];
        dir: string | null;

        config: Config | null;
    };
    export default project;
}

declare module "*.scene?raw" {
    const content: string;
    export default content;
}
