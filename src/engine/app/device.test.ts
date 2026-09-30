import { expect, test } from "bun:test";
import { stampAdapter, World } from "../index";
import { diagnose, loadScene, parseScene } from "../scene";

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
    const verdict = stampAdapter(fallbackAdapter);
    expect(verdict.class).toBe("fallback");
    expect(verdict.identity).toContain("SwiftShader");
});

test("an externally supplied GPU device without its adapter can be mistaken for a real adapter", () => {
    const verdict = stampAdapter();
    expect(verdict.class).toBe("unidentified");
    expect(verdict.identity).toBe("unidentified");
});

test("a CPU scene silently loses render-only attrs when those plugins are absent, so authors cannot see what the composition dropped", () => {
    const world = new World();
    const nodes = parseScene('<scene><a mesh="name: cube" material="name: default" /></scene>');
    const messages = diagnose(world, nodes);
    expect(messages.map((diagnostic) => diagnostic.message)).toEqual([
        '"mesh" has no active plugin registration; dropped',
        '"material" has no active plugin registration; dropped',
    ]);
    const result = loadScene(nodes, world);
    expect(result.dropped).toEqual(["mesh", "material"]);
    world.dispose();
});
