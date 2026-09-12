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

const cargoBuilds = new Map<string, string | null>();

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
    if (cached !== undefined) return cached;
    try {
        const proc = Bun.spawnSync(["cargo", "test", "--no-run", "-p", packageName], {
            cwd: root,
            stdout: "pipe",
            stderr: "pipe",
        });
        if (proc.success) {
            cargoBuilds.set(key, null);
            return null;
        }
        const detail = proc.stderr.toString().trim() || proc.stdout.toString().trim();
        const reason = `cargo test --no-run -p ${packageName} failed${detail ? `: ${detail}` : ""}`;
        cargoBuilds.set(key, reason);
        return reason;
    } catch (error) {
        const reason = `cargo is unavailable: ${(error as Error).message}`;
        cargoBuilds.set(key, reason);
        return reason;
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
