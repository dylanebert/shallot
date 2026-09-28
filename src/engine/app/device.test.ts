import { expect, test } from "bun:test";
import { Compute, State, stampAdapter } from "../index";
import { diagnose, load, parse } from "../scene";

const fallbackAdapter = {
    info: {
        vendor: "google",
        architecture: "swiftshader",
        device: "fallback",
        description: "SwiftShader",
        isFallbackAdapter: true,
    },
} as unknown as GPUAdapter;

test("GPU acquisition accepts a fallback adapter without stamping its verdict, so an app can look like it has real hardware", () => {
    stampAdapter(fallbackAdapter);
    expect(Compute.adapter.class).toBe("fallback");
    expect(Compute.adapter.identity).toContain("SwiftShader");
});

test("an externally supplied GPU device without its adapter can be mistaken for a real adapter", () => {
    stampAdapter();
    expect(Compute.adapter.class).toBe("unidentified");
    expect(Compute.adapter.identity).toBe("unidentified");
});

test("a CPU scene silently loses render-only attrs when those plugins are absent, so authors cannot see what the composition dropped", () => {
    const state = new State();
    const nodes = parse('<scene><a mesh="name: cube" material="name: default" /></scene>');
    const messages = diagnose(nodes);
    expect(messages.map((diagnostic) => diagnostic.message)).toEqual([
        '"mesh" has no active plugin registration; dropped',
        '"material" has no active plugin registration; dropped',
    ]);
    const result = load(nodes, state);
    expect(result.dropped).toEqual(["mesh", "material"]);
    state.dispose();
});
