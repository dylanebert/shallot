import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { relative, resolve } from "node:path";
import type { AdapterFacts } from "../engine/runtime/adapter";
import { launchPlan } from "./launch";
import { resolveSeat } from "./seat";

/** the result vocabulary printed by the surface reporter. */
export type VerdictResult = "pass" | "fail" | "refused" | "unrun";

/**
 * The host could not supply a row's premise, so the run never reached the predicate. Thrown only where the
 * premise itself is missing: the display seat, the page's presented rate, and a bounded call that runs out
 * of time before that rate is proved. Every other throw is a failure of the claim. The class travels as the
 * error's type, never as a prefix on its message, so no free-text string can promote a red to a refusal.
 */
export class MissingPremise extends Error {
    constructor(reason: string) {
        super(reason);
        this.name = "MissingPremise";
    }
}

/**
 * the reproduction record that travels with an integration verdict or refusal: enough to re-run the same
 * claim on the same seat, and enough to tell which seat it actually was. Runtime is telemetry here, never
 * a correctness floor.
 */
export interface Reproduction {
    host: string;
    launch: string;
    runtime: string;
    chromium: string;
    adapter: string;
    adapterClass: string;
    viewport: string;
    capture: string;
    engine: string;
    /** whether a real adapter is proven on this host or only declared; never grants a seat. */
    adapterEvidence?: string;
    /** the stepped tick the page asserted at, when the page reports one. */
    tick?: number;
}

/** the bounded diagnostics a failure retains. A pass retains none of them. */
export interface VerdictDiagnostics {
    pageErrors?: string[];
    gpuErrors?: string[];
    serverLog?: string;
    checks?: Array<{ name: string; detail?: string; data?: Record<string, number> }>;
    artifacts?: string[];
}

/** optional host measurements returned by an integration check. */
export interface VerdictMetadata {
    runtime?: string;
    hardware?: string;
    reason?: string;
    reproduction?: Reproduction;
    diagnostics?: VerdictDiagnostics;
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

// The probe reports the adapter's identity fields rather than its mere existence, because the seat policy
// has to tell a real device from a software fallback. `GPUAdapter.isFallbackAdapter` itself is not
// implemented by the Bun peer and throws, so only `info` is read.
const GPU_PROBE = `
const peer = await import("bun-webgpu");
await peer.setupGlobals();
const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) { console.log(JSON.stringify({ present: false })); process.exit(0); }
const info = adapter.info;
console.log(JSON.stringify({ present: true, info: {
    vendor: info?.vendor,
    architecture: info?.architecture,
    device: info?.device,
    description: info?.description,
    isFallbackAdapter: info?.isFallbackAdapter,
} }));
`;

function resolveGpuRequirement(root: string): string | null {
    if (gpuRequirement !== undefined) return gpuRequirement;
    const probe = Bun.spawnSync([process.execPath, "-e", GPU_PROBE], {
        cwd: root,
        stdout: "pipe",
        stderr: "pipe",
    });
    if (probe.exitCode !== 0) {
        const detail = `${probe.stderr.toString()}${probe.stdout.toString()}`.trim();
        gpuRequirement = `gpu seat unavailable: the WebGPU probe failed${detail ? `: ${detail}` : ""}`;
        return gpuRequirement;
    }
    let facts: AdapterFacts;
    try {
        facts = JSON.parse(probe.stdout.toString().trim().split("\n").at(-1) ?? "") as AdapterFacts;
    } catch (error) {
        gpuRequirement = `gpu seat unavailable: the WebGPU probe returned no adapter facts: ${(error as Error).message}`;
        return gpuRequirement;
    }
    const seat = resolveSeat("gpu", { device: facts });
    gpuRequirement = seat.ok ? null : seat.reason;
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

/** Compare `node --version` output with the exact version in `.node-version`. */
export function nodeVersionMismatch(pin: string, reported: string): string | null {
    if (!/^\d+\.\d+\.\d+$/.test(pin)) return `node pin must be an exact version, not ${pin}`;
    return reported.trim() === `v${pin}`
        ? null
        : `node ${reported.trim() || "(no version)"} does not match the pinned ${pin}`;
}

function resolveNode(root: string): string | null {
    const pinFile = resolve(root, ".node-version");
    if (!existsSync(pinFile)) return `node pin is unavailable at ${pinFile}`;
    try {
        const proc = Bun.spawnSync(["node", "--version"], { stdout: "pipe", stderr: "pipe" });
        if (!proc.success) return `node --version failed: ${proc.stderr.toString().trim()}`;
        return nodeVersionMismatch(readFileSync(pinFile, "utf8").trim(), proc.stdout.toString());
    } catch (error) {
        return `node is unavailable: ${(error as Error).message}`;
    }
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
        if (requirement === "node") {
            const reason = resolveNode(resolve(context.root ?? process.cwd()));
            if (reason !== null) return reason;
            continue;
        }
        if (requirement === "gpu") {
            const reason = resolveGpuRequirement(resolve(context.root ?? process.cwd()));
            if (reason !== null) return reason;
            continue;
        }
        if (requirement === "chromium") {
            return "chromium seat unavailable: retired until the device-seat roadmap item proves a real-adapter headless launch here";
        }
        if (requirement !== "display") return `runner cannot supply requirement ${requirement}`;
        // `display` is declared by the host that has one rather than inferred: a host with a window server
        // still runs every other row headlessly.
        if (!process.env.SHALLOT_DISPLAY_SEAT?.trim()) {
            const seat = resolveSeat("display", {});
            if (!seat.ok) return seat.reason;
        }
        // The launch path is the display seat's first premise: an undeclared host has no headed launch, and
        // no amount of installed Chromium substitutes for one. The seat itself resolves only on the adapter
        // its run observes.
        const plan = launchPlan(process.platform);
        if ("refused" in plan) return `display seat unavailable: ${plan.refused}`;
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

function objectField<T>(value: unknown): T | undefined {
    return value !== null && typeof value === "object" && !Array.isArray(value)
        ? (value as T)
        : undefined;
}

function metadataFrom(value: unknown): VerdictMetadata {
    if (value === null || typeof value !== "object") return {};
    const candidate = value as Record<string, unknown>;
    return {
        runtime: typeof candidate.runtime === "string" ? candidate.runtime : undefined,
        hardware: typeof candidate.hardware === "string" ? candidate.hardware : undefined,
        reason: typeof candidate.reason === "string" ? candidate.reason : undefined,
        reproduction: objectField<Reproduction>(candidate.reproduction),
        diagnostics: objectField<VerdictDiagnostics>(candidate.diagnostics),
    };
}

/** extract host labels from a returned value or an error carrying process-run diagnostics. */
export function verdictMetadata(value: unknown): VerdictMetadata {
    return metadataFrom(value);
}

/**
 * print the one-line, structured verdict for a non-step or refused check. A pass carries the compact
 * reproduction record; a failure or refusal also carries the bounded diagnostics behind it.
 */
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
        ...(metadata.reproduction === undefined ? {} : { reproduction: metadata.reproduction }),
        ...(result === "pass" || metadata.diagnostics === undefined
            ? {}
            : { diagnostics: metadata.diagnostics }),
    };
    console.log(`shallot verdict ${JSON.stringify(line)}`);
}
