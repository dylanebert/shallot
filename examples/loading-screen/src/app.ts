import type { Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { Welcome } from "./welcome";

export const AppPlugin: Plugin = {
    name: "LoadingScreenApp",
    dependencies: [Welcome, DiagnosticsOverlayPlugin],
};
