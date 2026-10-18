// Prints each warm window's unadmitted allocation rows of voxel turnover: sample count, bytes and the stack
// below the scheduler. argv: probe.entry.ts knobs, and `all` to print admitted wrapper rows too.
import { resolve } from "node:path";
import { sampleAllocation } from "../first-person-allocation/allocation";
import { isAdmittedAllocationSite } from "../first-person-allocation/allocation-admissions";

const words = process.argv.slice(2);
const sample = await sampleAllocation(resolve(import.meta.dir, "probe.entry.ts"), {
    warm: 2_400,
    frames: 60,
    input: words.filter((word) => word !== "all").join(" "),
    retainOptimizedSites: true,
});
for (const window of sample.windows) {
    console.log(`== ${window.label}; optimized ${window.optimizations?.join(", ") || "nothing"}`);
    for (const row of window.sites) {
        if (!words.includes("all") && isAdmittedAllocationSite(row)) continue;
        const stack = row.stack?.map((frame) => frame.site.replace(/^(\S+) src\//, "$1 ")) ?? [];
        const from = stack.findLastIndex((site) => /^runGroup /.test(site));
        console.log(`${String(row.count).padStart(5)} ${String(row.bytes).padStart(7)} B  ${stack.slice(from + 1).join(" <- ")}`);
    }
}
console.log(`control rows ${sample.control.length}`);
