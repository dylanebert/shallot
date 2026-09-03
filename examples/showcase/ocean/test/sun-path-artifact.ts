import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { basename, extname } from "node:path";
import { CAPTURE, SUN_FACING, sunDirection } from "../src/conditions";
import { analyze, type LookReading, load } from "./look.oracle";

export interface ArtifactPanel {
    role: "default" | "sun-facing" | "gold-t26" | "gold-t43";
    source: string;
    sha256: string;
    sourceWidth: number;
    sourceHeight: number;
    horizonRow: number;
    fadeExtentPerWaterHeight: number;
    specksPerMegapixelOfWater: number;
}

export interface SunPathArtifact {
    revision: "shallot-ocean-look/S12";
    cameraToSun: {
        defaultDegrees: number;
        sunFacingDegrees: number;
        declaredSunFacingDegrees: number;
    };
    displayedPanelSize: { width: 1280; height: 720 };
    panels: ArtifactPanel[];
    captions: {
        panelSha256: string;
        horizonRow: number;
        fadeExtentPerWaterHeight: number;
        specksPerMegapixelOfWater: number;
    }[];
}

function degrees(value: number): number {
    return (value * 180) / Math.PI;
}

function cameraToSun(yaw: number, pitch: number): number {
    const view = [
        -Math.sin(yaw) * Math.cos(pitch),
        Math.sin(pitch),
        -Math.cos(yaw) * Math.cos(pitch),
    ];
    const light = sunDirection(CAPTURE.sunAzimuthOffset, CAPTURE.sunElevation);
    const dot = view.reduce((sum, value, index) => sum - value * light[index]!, 0);
    return degrees(Math.acos(Math.max(-1, Math.min(1, dot))));
}

function normalized(
    reading: LookReading,
    width: number,
    height: number,
): Pick<ArtifactPanel, "horizonRow" | "fadeExtentPerWaterHeight" | "specksPerMegapixelOfWater"> {
    const waterHeight = height - reading.horizon.row;
    return {
        horizonRow: reading.horizon.row,
        fadeExtentPerWaterHeight: reading.duskBalance.fadeExtent / waterHeight,
        specksPerMegapixelOfWater:
            reading.lowerBandBrightSpecks / ((waterHeight * width) / 1_000_000),
    };
}

async function panel(role: ArtifactPanel["role"], source: string): Promise<ArtifactPanel> {
    const bytes = await readFile(source);
    const image = await load(source);
    return {
        role,
        source: basename(source),
        sha256: createHash("sha256").update(bytes).digest("hex"),
        sourceWidth: image.width,
        sourceHeight: image.height,
        ...normalized(analyze(image), image.width, image.height),
    };
}

export async function instrument(
    paths: [string, string, string, string],
): Promise<SunPathArtifact> {
    const roles: ArtifactPanel["role"][] = ["default", "sun-facing", "gold-t26", "gold-t43"];
    const panels = await Promise.all(paths.map((path, index) => panel(roles[index]!, path)));
    const artifact: SunPathArtifact = {
        revision: "shallot-ocean-look/S12",
        cameraToSun: {
            defaultDegrees: cameraToSun(CAPTURE.camera.yaw, CAPTURE.camera.pitch),
            sunFacingDegrees: cameraToSun(SUN_FACING.camera.yaw, SUN_FACING.camera.pitch),
            declaredSunFacingDegrees: degrees(CAPTURE.sunElevation),
        },
        displayedPanelSize: { width: 1280, height: 720 },
        panels,
        captions: panels.map(
            ({ sha256, horizonRow, fadeExtentPerWaterHeight, specksPerMegapixelOfWater }) => ({
                panelSha256: sha256,
                horizonRow,
                fadeExtentPerWaterHeight,
                specksPerMegapixelOfWater,
            }),
        ),
    };
    assertArtifact(artifact);
    return artifact;
}

export function assertArtifact(artifact: SunPathArtifact): void {
    if (
        Math.abs(
            artifact.cameraToSun.sunFacingDegrees - artifact.cameraToSun.declaredSunFacingDegrees,
        ) > 1e-9
    )
        throw new Error("sun-facing camera-to-sun relation differs from the declared condition");
    if (artifact.panels.length !== 4 || artifact.captions.length !== 4)
        throw new Error("artifact must retain two captures and two marked gold frames");
    if (new Set(artifact.panels.map(({ sha256 }) => sha256)).size !== 4)
        throw new Error("artifact panels must be byte-distinct");
    for (const [index, panel] of artifact.panels.entries()) {
        if (panel.sourceWidth <= 0 || panel.sourceHeight <= 0)
            throw new Error("panel dimensions must be positive");
        const caption = artifact.captions[index];
        if (
            !caption ||
            caption.panelSha256 !== panel.sha256 ||
            caption.horizonRow !== panel.horizonRow ||
            caption.fadeExtentPerWaterHeight !== panel.fadeExtentPerWaterHeight ||
            caption.specksPerMegapixelOfWater !== panel.specksPerMegapixelOfWater
        )
            throw new Error("caption readings must come from the panel source");
    }
}

function escapeHtml(value: string): string {
    return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

async function svg(
    paths: [string, string, string, string],
    artifact: SunPathArtifact,
): Promise<string> {
    const encoded = await Promise.all(
        paths.map(
            async (path) =>
                `data:image/${extname(path).toLowerCase() === ".png" ? "png" : "jpeg"};base64,${(await readFile(path)).toString("base64")}`,
        ),
    );
    const labels = artifact.panels.map((panel, index) => {
        const x = index * 1280;
        const caption = `${panel.role} | ${panel.sourceWidth}x${panel.sourceHeight} | sha256 ${panel.sha256} | horizon ${panel.horizonRow}/${panel.sourceHeight} | fade/water ${panel.fadeExtentPerWaterHeight.toFixed(4)} | specks/MP water ${panel.specksPerMegapixelOfWater.toFixed(2)}`;
        return `<image x="${x}" y="0" width="1280" height="720" preserveAspectRatio="none" href="${encoded[index]}"/><text x="${x + 20}" y="760">${escapeHtml(caption)}</text>`;
    });
    const relation = `camera-to-sun default ${artifact.cameraToSun.defaultDegrees.toFixed(3)}°; sun-facing ${artifact.cameraToSun.sunFacingDegrees.toFixed(3)}°; declared ${artifact.cameraToSun.declaredSunFacingDegrees.toFixed(3)}°`;
    return `<svg xmlns="http://www.w3.org/2000/svg" width="5120" height="838" viewBox="0 0 5120 838"><rect width="5120" height="838" fill="#111"/><g fill="#eee" font-family="monospace" font-size="17">${labels.join("")}<text x="20" y="810">${escapeHtml(relation)}</text></g></svg>\n`;
}

if (import.meta.main) {
    if (process.argv.length !== 8)
        throw new Error(
            "usage: bun sun-path-artifact.ts <default.png> <sun-facing.png> <t26.jpg> <t43.jpg> <artifact.svg> <manifest.json>",
        );
    const paths = process.argv.slice(2, 6) as [string, string, string, string];
    const output = process.argv[6]!;
    const manifest = process.argv[7]!;
    const artifact = await instrument(paths);
    await writeFile(output, await svg(paths, artifact));
    await writeFile(manifest, `${JSON.stringify(artifact, null, 2)}\n`);
    console.log(`artifact ${output} 5120x838`);
    console.log(`manifest ${manifest}`);
    console.log(
        `camera-to-sun default=${artifact.cameraToSun.defaultDegrees.toFixed(3)}° sun-facing=${artifact.cameraToSun.sunFacingDegrees.toFixed(3)}° declared=${artifact.cameraToSun.declaredSunFacingDegrees.toFixed(3)}°`,
    );
    for (const panel of artifact.panels)
        console.log(
            `${panel.role} ${panel.sourceWidth}x${panel.sourceHeight} sha256=${panel.sha256} horizon=${panel.horizonRow}/${panel.sourceHeight} fade/water=${panel.fadeExtentPerWaterHeight.toFixed(4)} specks/MPwater=${panel.specksPerMegapixelOfWater.toFixed(2)}`,
        );
}
