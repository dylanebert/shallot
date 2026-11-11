import { bunChild } from "../../scripts/bun-child";

const trials: Map<string, number>[] = [];
const expected = process.argv.includes("--focus") ? 6 : 146;
for (let trial = 0; trial < 3; trial++) {
    const result = Bun.spawnSync({
        ...bunChild([`${import.meta.dir}/decode.oracle.ts`, "--trial", String(trial), ...process.argv.slice(2)]),
        stdout: "pipe", stderr: "inherit",
    });
    const output = result.stdout.toString();
    process.stdout.write(output);
    if (result.exitCode) throw new Error(`meshopt trial ${trial} failed`);
    const ratios = new Map<string, number>();
    for (const line of output.split("\n")) {
        const fields = line.split("\t");
        if (fields.length === 5 && fields[0] !== "asset") ratios.set(`${fields[0]}:${fields[1]}`, Number(fields[4]));
    }
    if (ratios.size !== expected) throw new Error(`trial ${trial}: expected ${expected} recorded views, got ${ratios.size}`);
    trials.push(ratios);
}
const slower = [...trials[0].keys()].filter((key) => trials.every((trial) => trial.get(key)! > 1));
console.log("Slower in all three fresh processes:");
for (const key of slower) console.log(`${key}\t${trials.map((trial) => trial.get(key)).join("\t")}`);
console.log(`${slower.length} / ${expected} views slower in all three processes; timing is reported, not asserted.`);
