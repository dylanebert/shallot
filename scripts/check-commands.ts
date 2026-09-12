import { existsSync, readFileSync, realpathSync, statSync } from "node:fs";
import { Glob } from "bun";
import { dirname, relative, resolve } from "path";

async function readScripts(pkgPath: string): Promise<Record<string, string>> {
    const pkg = (await Bun.file(pkgPath).json()) as { scripts?: Record<string, string> };
    return pkg.scripts ?? {};
}

// Expand the root `workspaces` globs (Bun's own resolution: a literal directory, or `<base>/*`)
// into every member's package.json path.
async function workspacePkgPaths(rootDir: string, patterns: string[]): Promise<string[]> {
    const paths: string[] = [];
    for (const pattern of patterns) {
        const starIdx = pattern.indexOf("*");
        if (starIdx === -1) {
            paths.push(resolve(rootDir, pattern, "package.json"));
            continue;
        }
        const base = pattern.slice(0, starIdx).replace(/\/$/, "");
        const glob = new Glob(pattern.slice(base.length + 1));
        for await (const match of glob.scan({ cwd: resolve(rootDir, base), onlyFiles: false })) {
            paths.push(resolve(rootDir, base, match, "package.json"));
        }
    }
    return paths.filter((path) => existsSync(path));
}

// Every declared script resolves to an existing file or directory, or delegates to a script its
// target declares. `bunx` segments and bare `bun test` filters are external and unchecked.
async function checkExists(pkgPaths: string[]): Promise<{ detail: string }[]> {
    const violations: { detail: string }[] = [];
    for (const pkgPath of pkgPaths) {
        const dir = dirname(pkgPath);
        for (const [name, cmd] of Object.entries(await readScripts(pkgPath))) {
            for (const segment of cmd.split("&&")) {
                if (/\bbunx\s+/.test(segment)) continue;
                const match = segment.match(/\bbun\s+(?:(run|test)\s+)?(?:--cwd\s+(\S+)\s+)?(\S+)/);
                if (!match) continue;
                const [, verb, cwdArg, token] = match;
                if (token.startsWith("-")) continue;
                const base = cwdArg ? resolve(dir, cwdArg) : dir;
                if (token.includes("/") || token.includes(".")) {
                    const target = resolve(base, token);
                    if (!existsSync(target)) {
                        violations.push({
                            detail: `${pkgPath} ${name}: target "${token}" does not exist (resolved ${target})`,
                        });
                    }
                    continue;
                }
                if (verb === "test") continue;
                const targetPkgPath = resolve(base, "package.json");
                if (!existsSync(targetPkgPath)) continue;
                if (!(token in (await readScripts(targetPkgPath)))) {
                    violations.push({
                        detail: `${pkgPath} ${name}: delegates to script "${token}" not declared in ${targetPkgPath}`,
                    });
                }
            }
        }
    }
    return violations;
}

/** Engine `files` entries written at build or pack time: the tooling bundle. */
const PRODUCED = ["dist"];

/** Every declared bin and positive files entry must exist or have a pack producer. */
async function checkRealization(root: string): Promise<string[]> {
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
            dir === resolve(root) && kind === "files" && PRODUCED.includes(target);
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
                ) {
                    errors.push(
                        `${relative(root, manifest)} ${kind}: ${declared} is missing and has no pack-time projection`,
                    );
                }
            }
        }
    }
    return errors;
}

// Consumer commands use the installed bin; repository commands must resolve in this tree,
// without a global link or bunx downloading an unrelated registry version.
const root = resolve(import.meta.dir, "..");
const commandErrors = await checkRealization(root);
const entry = (await Bun.file(resolve(root, "CONTRIBUTING.md")).text())
    .split("## Commands\n")[1]
    ?.split("\n## ")[0];
if (!entry) commandErrors.push("CONTRIBUTING.md: missing Commands block");
const scripts = (await Bun.file(resolve(root, "package.json")).json()).scripts;
let inCommandFence = false;
let commandCount = 0;
for (const line of (entry ?? "").split("\n")) {
    if (line.startsWith("```")) {
        inCommandFence = !inCommandFence;
        continue;
    }
    if (!inCommandFence) continue;
    for (const segment of line.split("#")[0].split(/&&|;/)) {
        const command = segment.trim();
        if (!command || command.startsWith("#")) continue;
        commandCount++;
        const match = /^(bunx|bun)\s+(?:run\s+)?([^\s]+)/.exec(command);
        const token = match?.[2];
        let reachable = false;
        if (match?.[1] === "bunx") {
            const bin = resolve(root, "node_modules/.bin", token!);
            const expected = resolve(root, "bin/shallot.ts");
            reachable =
                token === "shallot" &&
                existsSync(bin) &&
                existsSync(expected) &&
                realpathSync(bin) === realpathSync(expected);
        } else if (token) {
            reachable =
                Object.hasOwn(scripts, token) ||
                (token.includes("/") && existsSync(resolve(root, token)));
        }
        if (!reachable)
            commandErrors.push(`CONTRIBUTING.md: unreachable repository command: ${command}`);
    }
}
if (!commandCount) commandErrors.push("CONTRIBUTING.md: empty command population");
commandErrors.push(
    ...(await checkExists([resolve(root, "package.json")])).map((error) => error.detail),
);
if (commandErrors.length) {
    console.error(`✗ command resolution:\n${commandErrors.join("\n")}`);
    process.exit(1);
}
console.log(
    `✓ command resolution: ${commandCount} repository commands; declared bin/files realization`,
);
