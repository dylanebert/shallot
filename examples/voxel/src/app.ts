import type { Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { TerrainPlugin } from "./terrain";

export const AppPlugin: Plugin = {
    name: "VoxelApp",
    dependencies: [TerrainPlugin, DiagnosticsOverlayPlugin],
};
