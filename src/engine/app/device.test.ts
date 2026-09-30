import { expect, test } from "bun:test";
import { stampAdapter } from "../index";

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
