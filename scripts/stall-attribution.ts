import { classifyCpuFrame, harnessBucketNames } from "../packages/shallot-cli/bin/verify";
import { skipReason, type VerifyResult, verify } from "./verify";

/** Assert the actual captured profile, not the classifier's fixtures. Named harness frames only:
 * anonymous readiness evaluation and the deliberate __shallotRafTick sampler remain blind spots. */
export function assertCapture(result: VerifyResult | null): void {
    if (result?.pass !== true)
        throw new Error(`verify failed: ${result?.error ?? "missing or unsuccessful result"}`);
    const ua = result.attribution?.userAgent;
    if (typeof ua !== "string" || !ua.trim()) throw new Error("missing observed user agent");
    if (ua.includes("HeadlessChrome")) throw new Error("attribution launched HeadlessChrome");
    const profile = result.cpuProfile;
    if (
        !profile ||
        !Number.isFinite(profile.totalMs) ||
        profile.totalMs <= 0 ||
        !Array.isArray(profile.entries) ||
        !profile.entries.length ||
        !Array.isArray(profile.buckets) ||
        !profile.buckets.length ||
        profile.entries.some(
            (entry) =>
                !entry ||
                typeof entry.functionName !== "string" ||
                typeof entry.url !== "string" ||
                !Number.isFinite(entry.selfMs) ||
                entry.selfMs <= 0,
        ) ||
        profile.buckets.some(
            (bucket) =>
                !bucket ||
                typeof bucket.name !== "string" ||
                !bucket.name.trim() ||
                !Number.isFinite(bucket.selfMs) ||
                bucket.selfMs <= 0,
        )
    )
        throw new Error("missing, empty or invalid captured CPU profile");
    // The predicate returns only buckets that exist in the capture. Missing bucket identities
    // must refuse, not hide a named contaminating entry behind a malformed summary.
    const names = new Set(
        profile.entries.map((entry) => classifyCpuFrame(entry.url, entry.functionName)),
    );
    if (
        names.size !== profile.buckets.length ||
        new Set(profile.buckets.map((bucket) => bucket.name)).size !== names.size ||
        profile.buckets.some((bucket) => !names.has(bucket.name))
    )
        throw new Error("invalid captured CPU profile bucket identities");
    const offending = harnessBucketNames(profile);
    if (offending.length) throw new Error(`harness contamination: ${offending.join(", ")}`);
    console.log(`No registered harness frames: ${profile.totalMs} sampled ms; observed UA ${ua}`);
}

/** By-path live driver; injected dependencies exercise refusal/exit propagation without a browser. */
export async function main(argv: string[], deps = { verify, skipReason }): Promise<number> {
    try {
        let dir = "examples/showcase/sandbox";
        const query: string[] = [];
        for (let i = 0; i < argv.length; i++) {
            const flag = argv[i];
            if (flag !== "--dir" && flag !== "--query") throw new Error(`unknown option: ${flag}`);
            const value = argv[++i];
            if (!value?.trim() || value.startsWith("--")) throw new Error(`missing value: ${flag}`);
            if (flag === "--dir") dir = value;
            else query.push("--query", value);
        }
        const reason = deps.skipReason();
        if (reason) throw new Error(`attribution requires headed display: ${reason}`);
        assertCapture(await deps.verify(dir, ["--attribution", "--timeout", "30000", ...query]));
        return 0;
    } catch (error) {
        console.error(error instanceof Error ? error.message : String(error));
        return 1;
    }
}

if (import.meta.main) process.exitCode = await main(process.argv.slice(2));
