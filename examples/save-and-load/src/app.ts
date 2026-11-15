import type { Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { Scene } from "./scene";

export const AppPlugin: Plugin = {
    name: "SaveAndLoadApp",
    dependencies: [Scene, DiagnosticsOverlayPlugin],
};
