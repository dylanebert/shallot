import type { Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { RoomPlugin } from "./room";

export const AppPlugin: Plugin = {
    name: "InteractApp",
    dependencies: [RoomPlugin, DiagnosticsOverlayPlugin],
};
