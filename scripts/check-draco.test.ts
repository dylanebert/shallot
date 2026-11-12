import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { DRACO_ARTIFACT, dracoArtifactIsFresh } from "../crates/draco/scripts/inputs";

test("checked Draco artifact matches its build inputs", () => {
    const fresh = dracoArtifactIsFresh(resolve(import.meta.dir, ".."));
    if (!fresh)
        throw new Error(
            `${DRACO_ARTIFACT}: build-input stamp missing or stale. Regenerate: bun crates/draco/scripts/build.ts`,
        );
    expect(fresh).toBe(true);
});
