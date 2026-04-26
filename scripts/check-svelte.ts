import { $, Glob } from "bun";

const filter = process.argv[2];
const glob = new Glob("**/svelte.config.{js,ts}");
const dirs = new Set<string>();

for await (const path of glob.scan({ cwd: import.meta.dir + "/..", onlyFiles: true })) {
    if (path.includes("node_modules")) continue;
    const dir = path.replace(/\/svelte\.config\.\w+$/, "");
    if (filter && !dir.includes(filter)) continue;
    dirs.add(dir);
}

let failed = false;
for (const dir of [...dirs].sort()) {
    const result = await $`bunx svelte-check --threshold warning`
        .cwd(`${import.meta.dir}/../${dir}`)
        .nothrow()
        .quiet();
    const output = result.text();
    const match = output.match(/(\d+) ERRORS (\d+) WARNINGS/);
    const errors = match ? parseInt(match[1]) : 0;
    const warnings = match ? parseInt(match[2]) : 0;

    if (errors > 0 || warnings > 0) {
        console.log(`\n${dir}:`);
        for (const line of output.split("\n")) {
            if (line.includes("ERROR") || line.includes("WARNING")) {
                const cleaned = line.replace(/^\d+ /, "");
                console.log(`  ${cleaned}`);
            }
        }
        failed = true;
    }
}

if (failed) {
    process.exit(1);
} else {
    console.log("svelte-check: all clean");
}
