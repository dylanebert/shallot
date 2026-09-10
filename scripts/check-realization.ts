import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { Glob } from "bun";
import { workspacePkgPaths } from "./check-scripts";

/** Engine `files` entries written at build or pack time: recipes by prepack, tooling bundles, audio wasm. */
const PRODUCED = ["examples", "dist", "rust/audio/pkg"];

/** Every declared bin and positive files entry must exist or have a pack producer. */
export async function checkRealization(root: string): Promise<string[]> {
    const pkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
    const manifests = [
        resolve(root, "package.json"),
        ...(await workspacePkgPaths(root, pkg.workspaces ?? [])),
    ];
    const errors: string[] = [];
    for (const manifest of manifests) {
        const dir = dirname(manifest);
        const value = JSON.parse(readFileSync(manifest, "utf8"));
        const bins: string[] =
            typeof value.bin === "string" ? [value.bin] : Object.values(value.bin ?? {});
        const files: string[] = value.files ?? [];
        const projected = (target: string, kind: "bin" | "files"): boolean =>
            dir === resolve(root, "packages/shallot") &&
            kind === "files" &&
            PRODUCED.includes(target);
        for (const [kind, targets] of [
            ["bin", bins],
            ["files", files.filter((file) => !file.startsWith("!"))],
        ] as const) {
            for (const declared of targets) {
                const target = declared.replace(/^\.\//, "");
                const path = resolve(dir, target);
                const present =
                    kind === "bin"
                        ? existsSync(path) && statSync(path).isFile()
                        : existsSync(path) ||
                          [...new Glob(target).scanSync({ cwd: dir })].length > 0;
                if (
                    (path !== dir && !path.startsWith(`${dir}/`)) ||
                    (!present && !projected(target, kind))
                )
                    errors.push(
                        `${relative(root, manifest)} ${kind}: ${declared} is missing and has no pack-time projection`,
                    );
            }
        }
    }
    return errors;
}

if (import.meta.main) {
    const errors = await checkRealization(resolve(import.meta.dir, ".."));
    if (errors.length) console.error(errors.join("\n"));
    else console.log("✓ declared bin/files realization");
    process.exit(errors.length ? 1 : 0);
}
