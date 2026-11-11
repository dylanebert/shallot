import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { BASISU_ARTIFACT, basisuArtifactIsFresh } from "../crates/basisu/scripts/inputs";

test("checked Basis Universal artifact matches its build inputs", () => {
    const fresh = basisuArtifactIsFresh(resolve(import.meta.dir, ".."));
    if (!fresh)
        throw new Error(
            `${BASISU_ARTIFACT}: build-input stamp missing or stale. Regenerate: bun crates/basisu/scripts/build.ts`,
        );
    expect(fresh).toBe(true);
});
