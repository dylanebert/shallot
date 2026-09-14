import { expect } from "bun:test";
import { check } from "./check";
import { compileWgsl } from "./wgsl";

check(
    "WGSL compiles through the native Bun GPU seam",
    {
        claim: "compileWgsl reports Dawn validation for malformed WGSL and accepts valid WGSL",
        size: "integration",
        requires: ["gpu"],
        subject: "src/harness/wgsl.test.ts",
    },
    async () => {
        const invalid = await compileWgsl("this is not WGSL");
        expect(invalid).toBeString();
        expect(invalid).toContain("WGSL");

        const valid = await compileWgsl("@compute @workgroup_size(1) fn main() {}");
        expect(valid).toBeNull();
    },
);
