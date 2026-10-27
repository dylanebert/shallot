type Stage = "vertex" | "fragment";
type Entry = {
    visibility?: readonly (Stage | "compute")[];
    storage?: unknown;
    uniform?: unknown;
    texture?: unknown;
    sampler?: unknown;
    storageTexture?: unknown;
};
const limits = {
    "storage buffers": 8,
    "uniform buffers": 12,
    "sampled textures": 16,
    samplers: 16,
    "storage textures": 4,
} as const;

/** @internal Refuse layouts exceeding WebGPU's default per-stage binding limits. Null positions
 * reserve binding numbers without consuming resources. */
export function validateBindingLimits(
    owner: string,
    layouts: readonly { readonly entries: Record<string, Entry | null> }[],
): void {
    for (const stage of ["vertex", "fragment"] as const) {
        const counts = {
            "storage buffers": 0,
            "uniform buffers": 0,
            "sampled textures": 0,
            samplers: 0,
            "storage textures": 0,
        };
        for (const layout of layouts) {
            for (const entry of Object.values(layout.entries)) {
                if (!entry || !(entry.visibility ?? ["vertex", "fragment"]).includes(stage))
                    continue;
                if ("storage" in entry) counts["storage buffers"]++;
                else if ("uniform" in entry) counts["uniform buffers"]++;
                else if ("texture" in entry) counts["sampled textures"]++;
                else if ("sampler" in entry) counts.samplers++;
                else if ("storageTexture" in entry) counts["storage textures"]++;
            }
        }
        for (const kind of Object.keys(limits) as (keyof typeof limits)[]) {
            if (counts[kind] > limits[kind])
                throw new Error(
                    `${owner} uses ${counts[kind]} ${stage} ${kind}; WebGPU's limit is ${limits[kind]}`,
                );
        }
    }
}
