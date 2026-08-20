// Env-driven override arms for a device reading that needs a different build-time constant without a
// source edit — the idiom `test/straightness.spec.ts`'s `ROADS_ARM_LABEL`/`ROADS_STRAIGHTNESS` set on the
// Node/Playwright side; these read the browser side, where only Vite-exposed `import.meta.env` reaches
// (Vite merges `VITE_`-prefixed shell env vars into it automatically, no config needed — the same env the
// `bunx shallot dev` child process, spawned by `playwright.config.ts`'s `webServer`, inherits). Optional
// chained + try/caught the same way shallot's own `devEnabled`
// (`@dylanebert/shallot/render/core`) reads `import.meta.env.DEV`, so `bun test` (no Vite, no
// `import.meta.env`) falls back cleanly instead of throwing.

function readEnv(key: string): string | undefined {
    try {
        const env = import.meta.env as Record<string, unknown> | undefined;
        const v = env?.[key];
        return typeof v === "string" ? v : undefined;
    } catch {
        return undefined;
    }
}

/** `envNumber("VITE_ROADS_RELIEF", 40)` — `fallback` when unset, not a real number, or `import.meta.env`
 *  is absent (bun test). */
export function envNumber(key: string, fallback: number): number {
    const raw = readEnv(key);
    if (raw === undefined) return fallback;
    const n = Number(raw);
    return Number.isFinite(n) ? n : fallback;
}

/** `envFlag("VITE_ROADS_NO_CUT")` — true only for `"1"`/`"true"`, false otherwise (unset, absent
 *  `import.meta.env`, or any other value). A manual-only diagnostic, not a gate control: `test/roads.spec.ts`
 *  Phase 4 asserts the corridor is flattened and runs before Phase 5's capture, so a no-cut run reds
 *  before any control frame is written — the flag cannot be emitted by the device gate. */
export function envFlag(key: string): boolean {
    const raw = readEnv(key);
    return raw === "1" || raw === "true";
}
