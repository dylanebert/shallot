import type { Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { TurntablePlugin } from "./turntable";

export const AppPlugin: Plugin = {
    name: "HudApp",
    dependencies: [TurntablePlugin, DiagnosticsOverlayPlugin],
};
