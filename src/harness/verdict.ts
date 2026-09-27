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

function objectField<T>(value: unknown): T | undefined {
    return value !== null && typeof value === "object" && !Array.isArray(value)
        ? (value as T)
        : undefined;
}

function metadataFrom(value: unknown): VerdictMetadata {
    if (value === null || typeof value !== "object") return {};
    const candidate = value as Record<string, unknown>;
    const diagnostics = objectField<VerdictDiagnostics>(candidate.diagnostics);
    const failedChecks =
        candidate.ok === false && Array.isArray(candidate.checks)
            ? candidate.checks.flatMap((check) => {
                  if (
                      check === null ||
                      typeof check !== "object" ||
                      (check as { ok?: unknown }).ok !== false ||
                      typeof (check as { name?: unknown }).name !== "string"
                  )
                      return [];
                  const { name, detail, data } = check as {
                      name: string;
                      detail?: unknown;
                      data?: unknown;
                  };
                  return [
                      {
                          name,
                          ...(typeof detail === "string" ? { detail } : {}),
                          ...(data !== null && typeof data === "object" && !Array.isArray(data)
                              ? { data: data as Record<string, number> }
                              : {}),
                      },
                  ];
              })
            : [];
    return {
        runtime: typeof candidate.runtime === "string" ? candidate.runtime : undefined,
        hardware: typeof candidate.hardware === "string" ? candidate.hardware : undefined,
        reason: typeof candidate.reason === "string" ? candidate.reason : undefined,
        reproduction: objectField<Reproduction>(candidate.reproduction),
        diagnostics:
            failedChecks.length === 0
                ? diagnostics
                : { ...diagnostics, checks: [...(diagnostics?.checks ?? []), ...failedChecks] },
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
    metadata: VerdictMetadata,
): void {
    const line = {
        claim,
        size,
        runtime: metadata.runtime,
        hardware: metadata.hardware,
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
