import { Camera, type World } from "@dylanebert/shallot";
import {
    ScreenSpaceAmbientOcclusion,
    ScreenSpaceAmbientOcclusionQualityLevel,
} from "@dylanebert/shallot/standard/rendering";

export const AO_LEVELS = ["Off", "Low", "Medium", "High", "Ultra", "Custom"] as const;
export type AoLevel = (typeof AO_LEVELS)[number];

export function addAmbientOcclusion(world: World, level: string): void {
    if (!AO_LEVELS.includes(level as AoLevel)) throw new Error(`Unknown SSAO quality: ${level}`);
    if (level === "Off") return;
    for (const camera of world.query([Camera])) {
        world.add(camera, ScreenSpaceAmbientOcclusion, {
            qualityLevel: ScreenSpaceAmbientOcclusionQualityLevel[level as Exclude<AoLevel, "Off">],
        });
    }
}
