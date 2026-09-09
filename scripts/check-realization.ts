import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { Glob } from "bun";
import {
    recipeRoot,
    runtimeRecord,
    runtimeRoots,
    toolingDist,
    toolingRoots,
} from "../packages/shallot/scripts/projections";
import { workspacePkgPaths } from "./check-scripts";

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
        const projected = (target: string, kind: "bin" | "files"): boolean => {
            if (dir !== resolve(root, "packages/shallot")) return false;
            if (target === runtimeRecord) return true;
            if (kind === "files" && [recipeRoot, toolingDist].includes(target)) return true;
            for (const [owner, roots] of [
                ["shallot-runtime", runtimeRoots],
                ["shallot-cli", toolingRoots],
            ] as const) {
                for (const entry of roots) {
                    if (kind === "files" && target === entry) return true;
                    if (target !== entry && !target.startsWith(`${entry}/`)) continue;
                    const source = target.startsWith("src/standard/tumble/engine/")
                        ? "shallot-tumble"
                        : owner;
                    const canonical = resolve(root, "packages", source, target);
                    if (existsSync(canonical) && (kind === "files" || statSync(canonical).isFile()))
                        return true;
                }
            }
            return false;
        };
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
