import { expect, test } from "bun:test";
import { resolve } from "node:path";
import {
    AUDIO_ARTIFACT,
    AUDIO_BUILD_COMMAND,
    audioArtifactIsFresh,
} from "../crates/audio/scripts/audio-inputs";

test("checked audio artifact matches its build inputs", () => {
    const root = resolve(import.meta.dir, "..");
    const fresh = audioArtifactIsFresh(root);
    if (!fresh) {
        console.error(
            `${AUDIO_ARTIFACT}: build-input stamp missing or stale. Regenerate: ${AUDIO_BUILD_COMMAND}`,
        );
    }
    expect(fresh).toBe(true);
});
