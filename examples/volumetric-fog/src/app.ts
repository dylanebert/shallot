import type { Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { MotesPlugin } from "./motes";

export const AppPlugin: Plugin = {
    name: "VolumetricFogApp",
    dependencies: [MotesPlugin, DiagnosticsOverlayPlugin],
};
