import { describe, expect, test } from "bun:test";
import { assertArtifact, type SunPathArtifact } from "./sun-path-artifact";

function artifact(): SunPathArtifact {
    const panels = ["default", "sun-facing", "gold-t26", "gold-t43"].map((role, index) => ({
        role: role as "default" | "sun-facing" | "gold-t26" | "gold-t43",
        source: `${role}.png`,
        sha256: String(index).repeat(64),
        sourceWidth: 1280,
        sourceHeight: 720,
        horizonRow: 210 + index,
        fadeExtentPerWaterHeight: 0.5 + index / 100,
        specksPerMegapixelOfWater: 100 + index,
        highlightThresholdPercentile: 0.99,
        clippingFraction: index / 100,
        glitterMeanChroma: 20 + index,
        contiguousRunBreakup: 0.5 + index / 100,
    }));
    return {
        revision: "shallot-ocean-look/S21a",
        cameraToSun: {
            defaultDegrees: 160,
            sunFacingDegrees: 18.334649444186343,
            declaredSunFacingDegrees: 18.334649444186343,
        },
        displayedPanelSize: { width: 1280, height: 720 },
        panels,
        captions: panels.map((panel) => ({
            panelSha256: panel.sha256,
            horizonRow: panel.horizonRow,
            fadeExtentPerWaterHeight: panel.fadeExtentPerWaterHeight,
            specksPerMegapixelOfWater: panel.specksPerMegapixelOfWater,
            highlightThresholdPercentile: panel.highlightThresholdPercentile,
            clippingFraction: panel.clippingFraction,
            glitterMeanChroma: panel.glitterMeanChroma,
            contiguousRunBreakup: panel.contiguousRunBreakup,
        })),
    };
}

describe("sun-path acceptance artifact", () => {
    test("retains distinct captures and references with source-bound captions", () => {
        expect(() => assertArtifact(artifact())).not.toThrow();
    });

    test("rejects a camera relation, duplicate panel, and stale caption", () => {
        const relation = artifact();
        relation.cameraToSun.sunFacingDegrees += 1;
        expect(() => assertArtifact(relation)).toThrow("camera-to-sun");

        const duplicate = artifact();
        duplicate.panels[1]!.sha256 = duplicate.panels[0]!.sha256;
        expect(() => assertArtifact(duplicate)).toThrow("byte-distinct");

        const caption = artifact();
        caption.captions[0]!.horizonRow++;
        expect(() => assertArtifact(caption)).toThrow("caption readings");

        const percentile = artifact();
        percentile.captions[0]!.highlightThresholdPercentile = 0.95;
        expect(() => assertArtifact(percentile)).toThrow("caption readings");
    });
});
