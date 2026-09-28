import { readFileSync } from "node:fs";
import { plugin } from "bun";
import typegpu from "unplugin-typegpu/bun";

// A Bun-linked package resolves peers from its real path. Route TypeGPU's runtime entries through
// the consumer's copy so a project import and the engine share one instance; otherwise use the engine peer.
let typegpuRoot = process.cwd();
let typegpuManifestPath: string;
try {
    typegpuManifestPath = Bun.resolveSync("typegpu/package.json", typegpuRoot);
} catch {
    typegpuRoot = import.meta.dir;
    typegpuManifestPath = Bun.resolveSync("typegpu/package.json", typegpuRoot);
}
const typegpuManifest = JSON.parse(readFileSync(typegpuManifestPath, "utf8"));
plugin({
    name: "shallot-typegpu-peer",
    setup(build) {
        for (const subpath of Object.keys(typegpuManifest.exports)) {
            if (subpath === "./package.json") continue;
            const specifier = subpath === "." ? "typegpu" : `typegpu/${subpath.slice(2)}`;
            build.module(specifier, async () => ({
                exports: await import(Bun.resolveSync(specifier, typegpuRoot)),
                loader: "object",
            }));
        }
    },
});

plugin(
    typegpu({
        // Keep dependency JavaScript out (the hook breaks picomatch's default export), while reaching
        // Shallot's source in node_modules as well as the project's own TypeScript and JavaScript.
        include:
            /^(?:.*\.(?:[cm]?ts|tsx)|(?:(?!.*[/\\]node_modules[/\\]).*|.*[/\\]node_modules[/\\]@dylanebert[/\\]shallot[/\\]src[/\\].*)\.(?:[cm]?js|jsx))$/,
    }),
);
