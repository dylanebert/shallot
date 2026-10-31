import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { GLTF_ARTIFACT, gltfArtifactIsFresh } from "../crates/gltf/scripts/inputs";

test("checked glTF artifact matches its build inputs", () => {
    const fresh = gltfArtifactIsFresh(resolve(import.meta.dir, ".."));
    if (!fresh)
        throw new Error(
            `${GLTF_ARTIFACT}: build-input stamp missing or stale. Regenerate: bun crates/gltf/scripts/build.ts`,
        );
    expect(fresh).toBe(true);
});
