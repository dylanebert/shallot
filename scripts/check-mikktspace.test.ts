import { expect, test } from "bun:test";
import { resolve } from "node:path";
import {
    MIKKTSPACE_ARTIFACT,
    mikktspaceArtifactIsFresh,
} from "../crates/mikktspace/scripts/inputs";

test("checked MikkTSpace artifact matches its build inputs", () => {
    const fresh = mikktspaceArtifactIsFresh(resolve(import.meta.dir, ".."));
    if (!fresh)
        throw new Error(
            `${MIKKTSPACE_ARTIFACT}: build-input stamp missing or stale. Regenerate: bun crates/mikktspace/scripts/build.ts`,
        );
    expect(fresh).toBe(true);
});
