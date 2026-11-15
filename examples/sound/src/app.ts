import type { Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { YardPlugin } from "./yard";

export const AppPlugin: Plugin = {
    name: "SoundApp",
    dependencies: [YardPlugin, DiagnosticsOverlayPlugin],
};
