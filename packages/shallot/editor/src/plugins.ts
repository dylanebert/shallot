import {
    TransformsPlugin,
    InputPlugin,
    ComputePlugin,
    ViewportPlugin,
    RenderPlugin,
    RasterPlugin,
    AudioPlugin,
    PhysicsPlugin,
    PlayerPlugin,
    RaytracingPlugin,
    TweenPlugin,
    LinesPlugin,
    ArrowsPlugin,
    TextPlugin,
    SkylabPlugin,
    type Plugin,
} from "@dylanebert/shallot";
import type { DiscoveredPlugin } from "./project";

function dp(plugin: Plugin): DiscoveredPlugin {
    return { name: plugin.name!, plugin };
}

export const STANDARD_PLUGINS: DiscoveredPlugin[] = [
    dp(TransformsPlugin),
    dp(InputPlugin),
    dp(ComputePlugin),
    dp(ViewportPlugin),
    dp(RenderPlugin),
    dp(RasterPlugin),
    dp(AudioPlugin),
    dp(PhysicsPlugin),
    dp(PlayerPlugin),
];

export const SHALLOT_PLUGINS: DiscoveredPlugin[] = [
    dp(RaytracingPlugin),
    dp(TweenPlugin),
    dp(LinesPlugin),
    dp(ArrowsPlugin),
    dp(TextPlugin),
    dp(SkylabPlugin),
];

export function pluginName(dp: DiscoveredPlugin): string {
    return dp.name;
}
