import type { Loading, Plugin } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import type { HostFrame } from "./host";
import { revealAfterFirstFrame } from "./reveal";
import { LoadingWorld } from "./scene";

export const AppPlugin: Plugin = {
    name: "EmbeddedSceneApp",
    dependencies: [LoadingWorld, DiagnosticsOverlayPlugin],
};

/** The static composition plus the reveal plugin bound to this page's host frame. */
export function pagePlugin(host: HostFrame, loading: Pick<Loading, "error">): Plugin {
    return {
        ...AppPlugin,
        dependencies: [...AppPlugin.dependencies!, revealAfterFirstFrame(host, loading)],
    };
}
