import type { Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { DeliveryPlugin } from "./delivery";

export const AppPlugin: Plugin = {
    name: "WorldUiApp",
    dependencies: [DeliveryPlugin, DiagnosticsOverlayPlugin],
};
