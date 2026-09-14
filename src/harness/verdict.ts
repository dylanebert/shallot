import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { relative, resolve } from "node:path";

/** the result vocabulary printed by the surface reporter. */
export type VerdictResult = "pass" | "fail" | "refused" | "unrun";

/** optional host measurements returned by an integration check. */
export interface VerdictMetadata {
    runtime?: string;
    hardware?: string;
    reason?: string;
}

interface QuarantineRow {
    file: string;
    claim: string;
    reason: string;
}

const require = createRequire(import.meta.url);
function projectRoot(): string {
    return resolve(process.env.SHALLOT_PROJECT_ROOT ?? process.cwd());
}

function readQuarantine(): QuarantineRow[] {
    const path = resolve(projectRoot(), "quarantine.json");
    if (!existsSync(path)) return [];
    try {
        const value = JSON.parse(readFileSync(path, "utf8"));
        if (!Array.isArray(value)) return [];
        return value.filter(
            (row): row is QuarantineRow =>
                row !== null &&
                typeof row === "object" &&
                typeof row.file === "string" &&
                typeof row.claim === "string" &&
                typeof row.reason === "string",
        );
    } catch {
        return [];
    }
}

/** return a quarantine reason for a runtime registration, or null when the row is live. */
export function quarantineReason(file: string, claim: string): string | null {
    const relativeFile = relative(projectRoot(), file).split("\\").join("/");
    return (
        readQuarantine().find((row) => row.file === relativeFile && row.claim === claim)?.reason ??
        null
    );
}

interface RequirementContext {
    root?: string;
    subjects?: readonly string[];
}

interface CargoBuild {
    executable?: string;
    artifacts?: readonly CargoArtifact[];
    reason?: string;
}

export interface CargoArtifact {
    target?: { kind?: string[]; name?: string; test?: boolean };
    profile?: { test?: boolean };
    executable?: string;
}

const CARGO_LIBRARY_TARGET_KINDS = new Set(["lib", "rlib", "cdylib", "staticlib", "proc-macro"]);

/** Select the one current libtest executable reported by Cargo's JSON artifact stream. */
export function selectCargoTestExecutable(
    packageName: string,
    artifacts: readonly CargoArtifact[],
): { executable?: string; reason?: string } {
    const expectedTarget = packageName.replaceAll("-", "_");
    const executables = [
        ...new Set(
            artifacts
                .filter(
                    (artifact) =>
                        artifact.target?.name === expectedTarget &&
                        artifact.target.test === true &&
                        artifact.profile?.test === true &&
                        artifact.target.kind?.some((kind) =>
                            CARGO_LIBRARY_TARGET_KINDS.has(kind),
                        ) &&
                        typeof artifact.executable === "string",
                )
                .map((artifact) => artifact.executable as string),
        ),
    ];
    if (executables.length !== 1) {
        return {
            reason:
                executables.length === 0
                    ? `cargo test --no-run -p ${packageName} produced no current libtest executable`
                    : `cargo test --no-run -p ${packageName} produced multiple or missing libtest executables`,
        };
    }
    if (!existsSync(executables[0])) {
        return {
            reason: `cargo test --no-run -p ${packageName} produced multiple or missing libtest executables`,
        };
    }
    return { executable: executables[0] };
}

/** Select named current integration-test executables from the same Cargo artifact stream. */
export function selectCargoTestTargetExecutables(
    packageName: string,
    targetNames: readonly string[],
    artifacts: readonly CargoArtifact[],
): { executables?: readonly string[]; reason?: string } {
    const selected = targetNames.map((targetName) => {
        const matches = [
            ...new Set(
                artifacts
                    .filter(
                        (artifact) =>
                            artifact.target?.name === targetName &&
                            artifact.target.kind?.includes("test") &&
                            artifact.target.test === true &&
                            artifact.profile?.test === true &&
                            typeof artifact.executable === "string",
                    )
                    .map((artifact) => artifact.executable as string),
            ),
        ];
        return { targetName, matches };
    });
    if (
        targetNames.length === 0 ||
        new Set(targetNames).size !== targetNames.length ||
        selected.some(({ matches }) => matches.length !== 1 || !existsSync(matches[0]))
    ) {
        return {
            reason: `cargo test --no-run -p ${packageName} produced missing, stale, or ambiguous named test executables`,
        };
    }
    return { executables: selected.map(({ matches }) => matches[0] as string) };
}

const cargoBuilds = new Map<string, CargoBuild>();
let gpuRequirement: string | null | undefined;

function resolveGpuRequirement(root: string): string | null {
    if (gpuRequirement !== undefined) return gpuRequirement;
    const probe = Bun.spawnSync(
        [
            process.execPath,
            "-e",
            `const peer = await import("bun-webgpu"); await peer.setupGlobals(); const adapter = await navigator.gpu?.requestAdapter(); if (!adapter) { console.error("no WebGPU adapter"); process.exit(2); }`,
        ],
        { cwd: root, stdout: "pipe", stderr: "pipe" },
    );
    if (probe.exitCode === 0) {
        gpuRequirement = null;
        return null;
    }
    const detail = `${probe.stderr.toString()}${probe.stdout.toString()}`.trim();
    gpuRequirement = `GPU seat unavailable${detail ? `: ${detail}` : ""}`;
    return gpuRequirement;
}

function cargoPackage(root: string, subjects: readonly string[]): string | null {
    if (subjects.length !== 1)
        return "cargo requirement needs exactly one subject naming a Cargo crate";
    const subject = subjects[0];
    if (subject.startsWith("/") || subject.includes(".."))
        return `cargo requirement has an invalid subject: ${subject}`;
    const manifest = resolve(root, subject, "Cargo.toml");
    if (!existsSync(manifest)) return `cargo manifest is unavailable at ${manifest}`;
    const source = readFileSync(manifest, "utf8");
    const packageName = source.match(/^name\s*=\s*"([^"]+)"\s*$/m)?.[1];
    return packageName === undefined
        ? `cargo manifest has no package name: ${manifest}`
        : packageName;
}

function resolveCargo(root: string, subjects: readonly string[]): string | null {
    const packageName = cargoPackage(root, subjects);
    if (packageName === null || packageName.startsWith("cargo ")) return packageName;
    const key = `${root}\u0000${packageName}`;
    const cached = cargoBuilds.get(key);
    if (cached !== undefined) return cached.reason ?? null;
    try {
        // Keep the compiler invocation outside the check body. The JSON artifact is also the
        // authority for the executable: guessing a target/debug/deps filename can select a stale
        // binary after a source change.
        const proc = Bun.spawnSync(
            ["cargo", "test", "--no-run", "-p", packageName, "--message-format=json"],
            {
                cwd: root,
                stdout: "pipe",
                stderr: "pipe",
            },
        );
        if (!proc.success) {
            const detail = proc.stderr.toString().trim() || proc.stdout.toString().trim();
            const reason = `cargo test --no-run -p ${packageName} failed${detail ? `: ${detail}` : ""}`;
            cargoBuilds.set(key, { reason });
            return reason;
        }
        const artifacts = proc.stdout
            .toString()
            .split("\n")
            .flatMap((line): CargoArtifact[] => {
                try {
                    const value = JSON.parse(line) as { reason?: string } & CargoArtifact;
                    return value.reason === "compiler-artifact" ? [value] : [];
                } catch {
                    return [];
                }
            });
        const selected = selectCargoTestExecutable(packageName, artifacts);
        if (selected.reason !== undefined) {
            cargoBuilds.set(key, { reason: selected.reason });
            return selected.reason;
        }
        cargoBuilds.set(key, { executable: selected.executable, artifacts });
        return null;
    } catch (error) {
        const reason = `cargo is unavailable: ${(error as Error).message}`;
        cargoBuilds.set(key, { reason });
        return reason;
    }
}

/** Return the current libtest executable after the untimed Cargo requirement is resolved. */
export function cargoTestExecutable(root: string, subject: string): string {
    const packageName = cargoPackage(root, [subject]);
    if (packageName === null || packageName.startsWith("cargo ")) {
        throw new Error(packageName ?? "missing Cargo package");
    }
    const reason = resolveCargo(root, [subject]);
    if (reason !== null) throw new Error(reason);
    const build = cargoBuilds.get(`${root}\u0000${packageName}`);
    if (build?.executable === undefined || !existsSync(build.executable)) {
        throw new Error(`cargo test -p ${packageName} has no current libtest executable`);
    }
    return build.executable;
}

/** Return named current integration-test executables from the untimed Cargo requirement. */
export function cargoTestTargetExecutables(
    root: string,
    subject: string,
    targetNames: readonly string[],
): readonly string[] {
    const packageName = cargoPackage(root, [subject]);
    if (packageName === null || packageName.startsWith("cargo ")) {
        throw new Error(packageName ?? "missing Cargo package");
    }
    const reason = resolveCargo(root, [subject]);
    if (reason !== null) throw new Error(reason);
    const build = cargoBuilds.get(`${root}\u0000${packageName}`);
    const selected = selectCargoTestTargetExecutables(
        packageName,
        targetNames,
        build?.artifacts ?? [],
    );
    if (selected.reason !== undefined) throw new Error(selected.reason);
    return selected.executables ?? [];
}

/** Resolve each named premise. Cargo compilation is a once-per-process, untimed prerequisite. */
export function missingRequirement(
    requirements: readonly string[],
    context: RequirementContext = {},
): string | null {
    for (const requirement of requirements) {
        if (requirement === "cargo") {
            const reason = resolveCargo(
                resolve(context.root ?? process.cwd()),
                context.subjects ?? [],
            );
            if (reason !== null) return reason;
            continue;
        }
        if (requirement === "gpu") {
            const reason = resolveGpuRequirement(resolve(context.root ?? process.cwd()));
            if (reason !== null) return reason;
            continue;
        }
        if (requirement !== "chromium") {
            return `runner cannot supply requirement ${requirement}`;
        }
        try {
            const module = require("playwright") as {
                chromium?: { executablePath?: () => string };
            };
            const executable = module.chromium?.executablePath?.();
            if (typeof executable !== "string" || !existsSync(executable)) {
                return `launchable Chromium is unavailable${executable ? ` at ${executable}` : ""}`;
            }
        } catch {
            return "launchable Chromium is unavailable: playwright is unavailable";
        }
    }
    return null;
}

/** the default runtime/hardware labels for checks that do not return host measurements. */
export function defaultMetadata(): VerdictMetadata {
    return { runtime: `bun ${Bun.version}`, hardware: "none" };
}

function metadataFrom(value: unknown): VerdictMetadata {
    if (value === null || typeof value !== "object") return {};
    const candidate = value as Record<string, unknown>;
    return {
        runtime: typeof candidate.runtime === "string" ? candidate.runtime : undefined,
        hardware: typeof candidate.hardware === "string" ? candidate.hardware : undefined,
        reason: typeof candidate.reason === "string" ? candidate.reason : undefined,
    };
}

/** extract host labels from a returned value or an error carrying process-run diagnostics. */
export function verdictMetadata(value: unknown): VerdictMetadata {
    return metadataFrom(value);
}

/** print the one-line, structured verdict for a non-step or refused check. */
export function emitVerdict(
    claim: string,
    size: string,
    started: number,
    result: VerdictResult,
    metadata: VerdictMetadata = {},
): void {
    const defaults = defaultMetadata();
    const line = {
        claim,
        size,
        runtime: metadata.runtime ?? defaults.runtime,
        hardware: metadata.hardware ?? defaults.hardware,
        duration: Number((performance.now() - started).toFixed(2)),
        result,
        ...(metadata.reason === undefined ? {} : { reason: metadata.reason }),
    };
    console.log(`shallot verdict ${JSON.stringify(line)}`);
}
