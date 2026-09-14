import { expect } from "bun:test";
import {
    assertCaptureGeometry,
    CAPTURE_CONTRACT,
    captureIdentityLabel,
    captureIdentityMatches,
} from "@dylanebert/shallot/harness/capture";
import { check } from "@dylanebert/shallot/harness/check";

check(
    "the capture contract is one fixed identity that refuses another geometry",
    {
        claim: "a surface at another size or scale captures anyway under the declared identity, so two seats' pixels would be compared as one contract",
        subject: "src/harness/capture.ts",
    },
    () => {
        // The contract is explicit in every field a consumer would otherwise leave implicit.
        expect(CAPTURE_CONTRACT).toEqual({
            width: 1280,
            height: 720,
            deviceScale: 1,
            surface: "final-canvas",
            encoding: "rgba8-tight",
        });
        expect(captureIdentityLabel(CAPTURE_CONTRACT)).toBe("final-canvas 1280x720@1 rgba8-tight");
        expect(() => assertCaptureGeometry(1280, 720)).not.toThrow();
        for (const [width, height] of [
            [1280, 719],
            [1281, 720],
            [640, 360],
            [0, 0],
        ]) {
            expect(() => assertCaptureGeometry(width, height)).toThrow("capture refused");
        }
        // Every field distinguishes an identity: a changed scale, surface or encoding is another contract,
        // not the same one at a different setting.
        expect(captureIdentityMatches(CAPTURE_CONTRACT, { ...CAPTURE_CONTRACT })).toBe(true);
        for (const changed of [
            { width: 640 },
            { height: 360 },
            { deviceScale: 2 },
            { surface: "offscreen" as unknown as typeof CAPTURE_CONTRACT.surface },
            { encoding: "png" as unknown as typeof CAPTURE_CONTRACT.encoding },
        ]) {
            expect(
                captureIdentityMatches(CAPTURE_CONTRACT, { ...CAPTURE_CONTRACT, ...changed }),
            ).toBe(false);
        }
    },
);
