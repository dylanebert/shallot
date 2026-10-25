import type { Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { ScenePlugin } from "./scene";

export const AppPlugin: Plugin = {
    name: "PostProcessingApp",
    dependencies: [ScenePlugin, DiagnosticsOverlayPlugin],
};
