import type { Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { GalleryPlugin } from "./gallery";

export const AppPlugin: Plugin = {
    name: "MaterialsApp",
    dependencies: [GalleryPlugin, DiagnosticsOverlayPlugin],
};
