import { expect, test } from "bun:test";
import { compileWgsl } from "./wgsl";

test("compileWgsl reports Dawn validation for malformed WGSL and accepts valid WGSL", async () => {
    const invalid = await compileWgsl("this is not WGSL");
    expect(invalid).toBeString();
    expect(invalid).toContain("WGSL");

    const valid = await compileWgsl("@compute @workgroup_size(1) fn main() {}");
    expect(valid).toBeNull();
}, 20_000);
