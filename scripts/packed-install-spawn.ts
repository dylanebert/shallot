import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join, resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import semver from "semver";
import { BUN_CHILD, bunCommand } from "./bun-child";

const INSTALL_TIMEOUT = 60_000;

export function spawnPackedInstall(
    command: string[],
    cwd: string,
    label: string,
    timeout = INSTALL_TIMEOUT,
): string {
    const result = Bun.spawnSync(command, {
        ...BUN_CHILD,
        env: { ...process.env },
        cwd,
        stdout: "pipe",
        stderr: "pipe",
        timeout,
        killSignal: "SIGKILL",
    });
    const output = `${result.stdout.toString()}${result.stderr.toString()}`;
    if (result.exitedDueToTimeout) {
        throw new Error(
            `${label} timed out after ${timeout} ms (last 8192 bytes per stream):\n` +
                `stdout:\n${result.stdout.subarray(-8192).toString()}\n` +
                `stderr:\n${result.stderr.subarray(-8192).toString()}`,
        );
    }
    if (result.exitCode !== 0) throw new Error(`${label} failed:\n${output}`);
    return output;
}

export function packedInstall(args: string[], cwd: string, label: string): string {
    const manifest = JSON.parse(readFileSync(join(cwd, "package.json"), "utf8"));
    const lock = Bun.JSONC.parse(readFileSync(join(import.meta.dir, "../bun.lock"), "utf8")) as {
        lockfileVersion: number;
        workspaces: Record<string, unknown>;
        packages: Record<string, unknown>;
    };
    if (lock.lockfileVersion !== 1) {
        throw new Error(`unsupported Bun lockfileVersion ${lock.lockfileVersion}`);
    }
    const workspace: Record<string, unknown> = { name: manifest.name };
    for (const key of [
        "dependencies",
        "devDependencies",
        "optionalDependencies",
        "peerDependencies",
    ]) {
        if (manifest[key]) workspace[key] = manifest[key];
    }
    lock.workspaces = { "": workspace };
    const spec =
        manifest.dependencies?.["@dylanebert/shallot"] ??
        manifest.devDependencies?.["@dylanebert/shallot"];
    if (typeof spec !== "string" || !spec.startsWith("file:")) {
        throw new Error(`${label}: expected a packed file: @dylanebert/shallot dependency`);
    }
    const tarball = resolve(cwd, spec.slice(5));
    const bytes = readFileSync(tarball);
    const archive = gunzipSync(bytes);
    let packed: ReturnType<typeof JSON.parse>;
    for (let offset = 0; offset + 512 <= archive.length; ) {
        const header = archive.subarray(offset, offset + 512);
        const name = header.subarray(0, 100).toString().split("\0")[0];
        const size = Number.parseInt(header.subarray(124, 136).toString(), 8);
        if (!name) break;
        if (!Number.isSafeInteger(size) || size < 0 || offset + 512 + size > archive.length) {
            throw new Error(`${label}: invalid packed tar header at ${offset}`);
        }
        if (name === "package/package.json") {
            packed = JSON.parse(archive.subarray(offset + 512, offset + 512 + size).toString());
            break;
        }
        offset += 512 + Math.ceil(size / 512) * 512;
    }
    if (!packed) throw new Error(`${label}: tarball has no package/package.json`);
    const metadata: Record<string, unknown> = {};
    for (const key of ["dependencies", "peerDependencies", "bin"]) {
        if (packed[key]) metadata[key] = packed[key];
    }
    const optionalPeers = Object.keys(packed.peerDependenciesMeta ?? {}).filter(
        (name) => packed.peerDependenciesMeta[name].optional,
    );
    if (optionalPeers.length) metadata.optionalPeers = optionalPeers;
    // Callers pack once into a fresh scratch path: Bun reuses stale content if that path is repacked.
    lock.packages["@dylanebert/shallot"] = [
        `@dylanebert/shallot@${tarball}`,
        metadata,
        `sha512-${createHash("sha512").update(bytes).digest("base64")}`,
    ];
    writeFileSync(join(cwd, "bun.lock"), `${JSON.stringify(lock, null, 2)}\n`);
    const output = spawnPackedInstall(
        bunCommand([args[0]!, "--offline", "--frozen-lockfile", "--verbose", ...args.slice(1)]),
        cwd,
        label,
    );
    checkInstalledRanges(cwd, label);
    return output;
}

function checkInstalledRanges(cwd: string, label: string): void {
    const seen = new Set<string>();
    function visit(directory: string): void {
        const real = realpathSync(directory);
        if (seen.has(real)) return;
        seen.add(real);
        const path = join(real, "package.json");
        if (existsSync(path)) {
            const pkg = JSON.parse(readFileSync(path, "utf8"));
            const require = createRequire(path);
            const edges = [
                ...Object.entries(pkg.dependencies ?? {}).map(([name, spec]) => [
                    "dependency",
                    name,
                    spec,
                ]),
                ...Object.entries(pkg.peerDependencies ?? {})
                    .filter(([name]) => !pkg.peerDependenciesMeta?.[name]?.optional)
                    .map(([name, spec]) => ["peer", name, spec]),
            ] as [string, string, string][];
            for (const [kind, name, spec] of edges) {
                const target = require.resolve
                    .paths(name)
                    ?.map((base) => join(base, name, "package.json"))
                    .find(existsSync);
                const installed = target ? JSON.parse(readFileSync(target, "utf8")) : undefined;
                let range = spec;
                if (spec.startsWith("npm:")) {
                    const alias = spec.slice(4).match(/^(@[^/]+\/[^@]+|[^@]+)(?:@(.+))?$/);
                    if (!alias || installed?.name !== alias[1]) {
                        throw new Error(
                            `${label}: ${pkg.name} ${kind} ${name} range ${spec} installed ${installed?.version ?? "missing"}: invalid npm alias target`,
                        );
                    }
                    range = alias[2] ?? "*";
                }
                // The scratch project's newly packed engine is the only non-registry edge admitted.
                if (
                    spec.startsWith("file:") &&
                    real === realpathSync(cwd) &&
                    name === "@dylanebert/shallot" &&
                    installed?.name === name
                )
                    continue;
                const validRange = semver.validRange(range);
                if (validRange === null) {
                    throw new Error(
                        `${label}: ${pkg.name} ${kind} ${name} range ${spec} installed ${installed?.version ?? "missing"}: unsupported spec`,
                    );
                }
                if (!installed || !semver.satisfies(installed.version, validRange)) {
                    throw new Error(
                        `${label}: ${pkg.name} ${kind} ${name} range ${spec} installed ${installed?.version ?? "missing"}: unmet or unsupported range`,
                    );
                }
            }
        }
        const modules = join(real, "node_modules");
        if (existsSync(modules)) walk(modules);
    }
    function walk(directory: string): void {
        for (const entry of readdirSync(directory, { withFileTypes: true })) {
            if (!entry.isDirectory() && !entry.isSymbolicLink()) continue;
            const path = join(directory, entry.name);
            if (entry.name.startsWith("@") || entry.name === ".bun") walk(path);
            else if (!entry.name.startsWith(".")) visit(path);
        }
    }
    visit(cwd);
}
