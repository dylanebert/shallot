import { expect } from "bun:test";
import { resolve } from "node:path";
import { State } from "../engine";
import { check } from "./check";
import { installHarness } from "./runtime";

check(
    "the browser-facing harness entry excludes host-only runner modules",
    {
        claim: "the browser-facing harness entry does not import Bun or Node-only check registration and prerequisite modules",
    },
    async () => {
        const built = await Bun.build({
            entrypoints: [resolve(import.meta.dir, "index.ts")],
            metafile: true,
            target: "browser",
        });
        if (!built.success || built.metafile === undefined)
            throw new Error(`browser harness graph failed: ${built.logs.map(String).join("\\n")}`);
        const inputs = Object.keys(built.metafile.inputs).map((path) => resolve(path));
        for (const owner of [
            "check",
            "requirements",
            "quarantine",
            "allocation",
            "browser",
            "launch",
        ])
            expect(inputs.some((path) => path.endsWith(`/src/harness/${owner}.ts`))).toBe(false);
        expect(
            Object.values(built.metafile.inputs)
                .flatMap((input) => input.imports)
                .some((edge) => /^(?:node:|bun:)/.test(edge.path)),
        ).toBe(false);
    },
);

check(
    "the default browser target exposes only protocol outcomes",
    {
        claim: "the default browser protocol does not interpret pose or transform components, leaving domain assertions to product observation",
    },
    async () => {
        const state = new State();
        const previous = (globalThis as unknown as Window).__harness;
        try {
            const target = installHarness(state);
            expect(target.ready).toBe(false);
            expect("read" in target).toBe(false);
            await expect(target.run?.()).resolves.toEqual({
                ok: true,
                checks: [{ name: "booted", ok: true }],
            });
        } finally {
            state.dispose();
            (globalThis as unknown as Window).__harness = previous;
        }
    },
);
