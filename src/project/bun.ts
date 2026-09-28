import { existsSync, readFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import type { BunPlugin } from "bun";
import typegpu from "unplugin-typegpu/bun";

const PACKAGE_ROOT = resolve(import.meta.dir, "../..");
const SHALLOT = JSON.parse(readFileSync(join(PACKAGE_ROOT, "package.json"), "utf8"));
const TYPEGPU_RANGE = SHALLOT.peerDependencies.typegpu as string;
const TYPEGPU_INCLUDE =
    /^(?:.*\.(?:[cm]?ts|tsx)|(?:(?!.*[/\\]node_modules[/\\]).*|.*[/\\]node_modules[/\\]@dylanebert[/\\]shallot[/\\]src[/\\].*)\.(?:[cm]?js|jsx))$/;

function nearestPackageRoot(from: string): string | undefined {
    let dir = resolve(from);
    while (true) {
        if (existsSync(resolve(dir, "package.json"))) return dir;
        const parent = dirname(dir);
        if (parent === dir) return undefined;
        dir = parent;
    }
}

function typegpuRootFrom(file: string): string | undefined {
    const normalized = resolve(file);
    const marker = `${sep}node_modules${sep}typegpu${sep}`;
    const markerAt = normalized.lastIndexOf(marker);
    return markerAt < 0 ? undefined : normalized.slice(0, markerAt + marker.length - 1);
}

/** Bun's test preload: TypeGPU transform plus linked-peer dedupe for the consumer's compatible copy. */
export function shallot(): BunPlugin {
    // Bun exposes cwd and the entrypoint, but no project-root API. Walk up from cwd so tests launched
    // in a subdirectory still find the consumer rather than the linked engine's real path.
    const projectRoot = nearestPackageRoot(process.cwd()) ?? process.cwd();
    let projectTypegpuManifestPath: string | undefined;
    try {
        projectTypegpuManifestPath = Bun.resolveSync("typegpu/package.json", projectRoot);
    } catch {
        // No project copy: let Bun resolve the engine's peer normally.
    }

    let projectTypegpuRoot: string | undefined;
    let projectVersion: string | undefined;
    if (projectTypegpuManifestPath) {
        const manifest = JSON.parse(readFileSync(projectTypegpuManifestPath, "utf8"));
        const version = (projectVersion = manifest.version ?? "<unknown>");
        if (!Bun.semver.satisfies(version, TYPEGPU_RANGE)) {
            throw new Error(
                `Shallot requires TypeGPU ${TYPEGPU_RANGE}, but the project has TypeGPU ${projectVersion} at ${projectTypegpuManifestPath}.`,
            );
        }
        projectTypegpuRoot = dirname(projectTypegpuManifestPath);
    }

    const transform = typegpu({ include: TYPEGPU_INCLUDE });
    return {
        name: "shallot",
        async setup(build) {
            if (projectTypegpuManifestPath && projectTypegpuRoot && projectVersion) {
                // Bun reports linked TypeGPU's internal JS imports as relative paths from their real importer,
                // so redirect those along with the package entrypoints.
                build.onResolve({ filter: /typegpu|\.js$/ }, ({ path, importer }) => {
                    const importerRoot = importer ? typegpuRootFrom(importer) : undefined;
                    if (importerRoot && (path.startsWith("./") || path.startsWith("../"))) {
                        if (importerRoot === projectTypegpuRoot) return;
                        const subpath = relative(importerRoot, resolve(dirname(importer), path));
                        if (subpath.startsWith("..")) return;
                        const target = resolve(projectTypegpuRoot, subpath);
                        if (!existsSync(target)) {
                            throw new Error(
                                `Shallot's TypeGPU peer ${TYPEGPU_RANGE} cannot resolve ${path} from ${importer} in project TypeGPU ${projectVersion} at ${projectTypegpuManifestPath}.`,
                            );
                        }
                        return { path: target };
                    }

                    if (path === "typegpu" || path.startsWith("typegpu/")) {
                        return { path: Bun.resolveSync(path, projectRoot) };
                    }

                    const sourceRoot = typegpuRootFrom(path);
                    if (!sourceRoot || sourceRoot === projectTypegpuRoot) return;
                    const target = resolve(projectTypegpuRoot, relative(sourceRoot, resolve(path)));
                    if (!existsSync(target)) {
                        throw new Error(
                            `Shallot's TypeGPU peer ${TYPEGPU_RANGE} cannot resolve ${relative(sourceRoot, resolve(path))} in project TypeGPU ${projectVersion} at ${projectTypegpuManifestPath}.`,
                        );
                    }
                    return { path: target };
                });
            }
            await transform.setup(build);
        },
    };
}
