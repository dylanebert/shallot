import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
    windowBytes,
    type AllocationSite,
} from "../first-person-allocation/allocation";

const WARM_FRAMES = 12_000;
const SAMPLE_FRAMES = 600;
const runtimeSource = "src/standard/audio/runtime.ts";
const workletSource = "shallot-audio-worklet.js";

function hasFrame(site: AllocationSite, source: string): boolean {
    return site.site.includes(source) || site.stack?.some((frame) => frame.site.includes(source)) === true;
}

function printSites(label: string, sites: readonly AllocationSite[]): void {
    console.info(`${label}: ${windowBytes({ sites })} B at ${sites.length} sites`);
    for (const site of sites.slice(0, 12)) {
        const stack = site.stack?.map((frame) => `${frame.functionName} ${frame.site}`).join(" <- ");
        console.info(`  ${site.bytes} B / ${site.count} samples at ${site.site}${stack ? `\n    ${stack}` : ""}`);
    }
}

const sample = await sampleAllocation(resolve(import.meta.dir, "allocation.entry.ts"), {
    warm: WARM_FRAMES,
    frames: SAMPLE_FRAMES,
});

console.info(`${sample.runtime}; warm ${sample.warm}, sampled ${sample.frames} frames per window`);
const failures: string[] = [];
for (const window of sample.windows) {
    const runtimeSites = window.sites.filter((site) => hasFrame(site, runtimeSource));
    const processSites = window.sites.filter((site) => hasFrame(site, workletSource));
    console.info(
        `${window.label}: SoundSystem/runtime ${windowBytes({ sites: runtimeSites })} B; process() ${windowBytes({ sites: processSites })} B`,
    );
    if (window.optimizations?.length)
        failures.push(
            `${window.label}: sampled before optimization finished (${window.optimizations.join(", ")})`,
        );
    if (runtimeSites.length > 0)
        failures.push(`${window.label}: SoundSystem/runtime sampled heap was nonzero`);
    if (processSites.length > 0)
        failures.push(`${window.label}: process() sampled heap was nonzero`);
    if (runtimeSites.length > 0) printSites(`${window.label} SoundSystem/runtime sites`, runtimeSites);
    if (processSites.length > 0) printSites(`${window.label} process() sites`, processSites);
}

const controlBytes = windowBytes({ sites: sample.control });
printSites("one-allocation-per-frame control", sample.control);
if (controlBytes === 0) failures.push("allocation control was invisible to the sampler");

const otherSites = sample.windows.flatMap((window) =>
    window.sites.filter((site) => !hasFrame(site, runtimeSource) && !hasFrame(site, workletSource)),
);
if (otherSites.length > 0) printSites("other sampled sites (outside the audio runtime and processor)", otherSites);

const targetSample = {
    ...sample,
    windows: sample.windows.map((window) => ({
        ...window,
        sites: window.sites.filter(
            (site) => hasFrame(site, runtimeSource) || hasFrame(site, workletSource),
        ),
    })),
};
const allocation = allocationFailure(targetSample);
if (allocation) failures.push(allocation);

const optimizedWindow = sample.windows.some((window) => window.optimizations?.length);
const runtimeBytes = sample.windows.reduce(
    (total, window) => total + windowBytes({ sites: window.sites.filter((site) => hasFrame(site, runtimeSource)) }),
    0,
);
const processBytes = sample.windows.reduce(
    (total, window) => total + windowBytes({ sites: window.sites.filter((site) => hasFrame(site, workletSource)) }),
    0,
);
console.info(
    `audio allocation oracle: SoundSystem/runtime ${optimizedWindow || runtimeBytes > 0 ? "FAILED" : "0 B"}; process() ${optimizedWindow || processBytes > 0 ? "FAILED" : "0 B"}; control ${controlBytes} B`,
);
if (failures.length > 0) throw new Error(failures.join("\n"));
