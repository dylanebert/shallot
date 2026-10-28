import type { Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { IsometricPlugin } from "./isometric";

export const AppPlugin: Plugin = {
    name: "IsometricApp",
    dependencies: [IsometricPlugin, DiagnosticsOverlayPlugin],
};
