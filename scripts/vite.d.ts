import type { Plugin, Rollup } from "vite";

export declare const CROSS_ORIGIN_ISOLATION: {
    "Cross-Origin-Opener-Policy": string;
    "Cross-Origin-Embedder-Policy": string;
};

export declare function pluginPackages(projectDir?: string): string[];
export declare function findPublicDirs(projectDir: string): string[];
export declare function assetSrc(file: string, publicDirs: string[]): string | null;
export declare function orphanedAssets(bundle: Rollup.OutputBundle): string[];
export declare function classifyProjectFile(
    file: string,
    absDir: string,
    publicDirs: string[],
): "asset" | "project" | null;
export declare function shallot(projectDir?: string): Plugin[];

export declare function manifestPath(dir: string): string;
export declare function manifestWarnings(raw: string, known: ReadonlySet<string>): string[];
export declare function discoverScenes(dir: string): string[];
