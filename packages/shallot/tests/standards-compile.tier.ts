import { describe, expect, test } from "bun:test";
import { resolvedKernels, STANDARDS_POPULATION_GOLDEN } from "./standards";

// These exported functions are declaration-bound by their production consumers. Resolving them bare
// intentionally omits exactly one resource; the manifest makes that boundary closed and reviewable.
const CONSUMER_BOUND_RESOURCES = {
    bvhAnyHit: "nodes",
    bvhClosestHit: "nodes",
    collideHull: "hullData",
    collideRoundedPolytope: "hullData",
    lightFactor: "pointShadows",
    lit: "pointShadows",
    litPbr: "pointShadows",
    pointFactor: "pointShadows",
    polyMake: "hullData",
    sampleSunShadow: "sunShadow",
} as const satisfies Record<string, string>;

function unresolvedResources(message: string): string[] {
    return [...message.matchAll(/unresolved (?:identifier|value) '([^']+)'/g)].map(
        ([, resource]) => resource,
    );
}

// By-path full-corpus Tint compilation tier. Run from the shallot root:
// `bun test ./packages/shallot/tests/standards-compile.tier.ts`.
// Trigger cone: `packages/shallot/src/**/*.ts` and `packages/shallot/tests/standards*`; the source walk
// and its exact population golden make every resolved-kernel addition, removal, or WGSL change relevant.
// Red proof (2026-08-30): seeding standalone `applyGrade` with an unresolved `s4_missing` reference
// made this command exit 1 with a non-null GPUValidationError; restoring the seed exited 0.

describe("resolved kernel Tint compilation", () => {
    test("the complete standards population has exactly its declared Tint result", async () => {
        const kernels = await resolvedKernels();
        expect(kernels).toHaveLength(STANDARDS_POPULATION_GOLDEN);
        expect(Object.keys(CONSUMER_BOUND_RESOURCES)).toHaveLength(10);

        const adapter = await navigator.gpu.requestAdapter();
        expect(adapter).not.toBeNull();
        if (!adapter) return;
        const device = await adapter.requestDevice();

        const failures: string[] = [];
        const seenConsumerBound = new Set<string>();
        for (const kernel of kernels) {
            device.pushErrorScope("validation");
            device.createShaderModule({
                label: `standards:${kernel.name}`,
                code: kernel.wgsl,
            });
            const error = await device.popErrorScope();
            const expectedResource =
                CONSUMER_BOUND_RESOURCES[kernel.name as keyof typeof CONSUMER_BOUND_RESOURCES];

            if (!expectedResource) {
                if (error !== null) {
                    failures.push(
                        `${kernel.name} (${kernel.file}) expected null, got ${error.constructor.name}: ${error.message}`,
                    );
                }
                continue;
            }

            seenConsumerBound.add(kernel.name);
            if (!(error instanceof GPUValidationError)) {
                failures.push(
                    `${kernel.name} (${kernel.file}) expected GPUValidationError for ${expectedResource}, got ${error === null ? "null" : error.constructor.name}`,
                );
                continue;
            }
            const resources = unresolvedResources(error.message);
            if (resources.length !== 1 || resources[0] !== expectedResource) {
                failures.push(
                    `${kernel.name} (${kernel.file}) expected sole unresolved resource ${expectedResource}, got [${resources.join(", ")}]: ${error.message}`,
                );
            }
        }
        device.destroy();

        const manifestNames = Object.keys(CONSUMER_BOUND_RESOURCES).sort();
        const seenNames = [...seenConsumerBound].sort();
        if (seenNames.join("\0") !== manifestNames.join("\0")) {
            failures.push(
                `consumer-bound manifest drift: expected [${manifestNames.join(", ")}], saw [${seenNames.join(", ")}]`,
            );
        }
        expect(failures).toEqual([]);
    });
});
