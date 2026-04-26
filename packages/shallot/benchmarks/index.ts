import { runCoreBenchmarks } from "./core.bench";
import { runTransformsBenchmarks } from "./transforms.bench";
import { runMathBenchmarks } from "./math.bench";
import { runCapacityBenchmarks } from "./capacity.bench";
import { runEventsBenchmarks } from "./events.bench";

async function main() {
    console.log("╔════════════════════════════════════════╗");
    console.log("║       shallot performance benchmarks   ║");
    console.log("╚════════════════════════════════════════╝");

    await runMathBenchmarks();
    await runCapacityBenchmarks();
    await runCoreBenchmarks();
    await runTransformsBenchmarks();
    await runEventsBenchmarks();
}

main().catch(console.error);
