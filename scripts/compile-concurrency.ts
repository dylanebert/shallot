import { PIPELINE_COMPILE_MEASURE_PREFIX } from "../packages/shallot/src/engine/runtime/gpu";
import { compileConcurrencyRatio } from "../site/rum-compile-vitals";
import { queryFlags, skipReason, teardownBridge, verify } from "./verify";

// The permanent capture of the boot pipeline-compile chain's own concurrency: nothing else in the
// tree reads the raw `PIPELINE_COMPILE_MEASURE_PREFIX`-filtered `performance.measure` entries out of
// a real run. Re-runnable, the same shape as `scripts/stall-attribution.ts`: one `shallot verify
// --attribution` boot (`bin/verify.ts`'s `attribution.compileMeasures`, every raw `measure` entry over
// the boot window, unfiltered), fed through the pure reader `compileConcurrencyRatio`
// (`site/rum-compile-vitals.ts`) rather than re-deriving the ratio here. Prints
// `{count, sum, span, ratio}` plus the absolute span ungated — a concurrency claim (the ratio) is
// host-independent by construction, so nothing here gates on a wall-clock number.
//
//     bun run scripts/compile-concurrency.ts [--dir <project>] [--query k=v ...]
//
// Default `--dir` is `examples/showcase/sandbox` — the project the recorded baseline ran against.

interface Args {
    dir: string;
    query: string[];
}

function parseArgs(argv: string[]): Args {
    const out: Args = { dir: "examples/showcase/sandbox", query: [] };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === "--dir") out.dir = argv[++i] ?? out.dir;
        else if (arg === "--query") out.query.push(argv[++i] ?? "");
        else throw new Error(`unknown option: ${arg}`);
    }
    return out;
}

async function main(): Promise<void> {
    const args = parseArgs(process.argv.slice(2));
    const reason = skipReason();
    if (reason) {
        console.log(`skip: ${reason}`);
        return;
    }

    console.log(
        `booting ${args.dir} (--attribution) to read the boot pipeline-compile concurrency ratio…`,
    );
    const result = await verify(args.dir, [
        "--attribution",
        "--timeout",
        "30000",
        ...queryFlags(args.query),
    ]);

    if (!result) {
        console.log("no result — verify crashed before reporting (see its own stderr above)");
        process.exitCode = 1;
        return;
    }
    if (!result.pass) {
        console.log(`verify did not pass: ${result.error ?? JSON.stringify(result.errors)}`);
    }
    const attribution = result.attribution;
    if (!attribution) {
        console.log(
            "no attribution on the result — --attribution wasn't honored, or the page.evaluate " +
                "read failed (see verify's own errors above)",
        );
        process.exitCode = 1;
        return;
    }

    const { count, sum, span, ratio } = compileConcurrencyRatio(
        attribution.compileMeasures,
        PIPELINE_COMPILE_MEASURE_PREFIX,
    );

    console.log("\n## S2 — boot pipeline-compile concurrency ratio\n");
    console.log(`adapter: ${result.hardware ?? "unknown"}`);
    console.log(`count: ${count}`);
    console.log(`sum: ${sum.toFixed(1)}ms`);
    console.log(`span: ${span.toFixed(1)}ms`);
    console.log(`ratio: ${ratio.toFixed(3)}`);
    console.log(
        `\nratio ≈ 1.0 reads as fully serial; ratio > 1 reads as N-way overlap. The absolute span is ` +
            `printed ungated — it is a wall-clock reading, not the structural claim (\`compileConcurrencyRatio\`'s own docblock).`,
    );

    await teardownBridge();
}

await main();
