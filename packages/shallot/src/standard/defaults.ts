import { setDefaultPlugins, setDefaultLoading } from "../engine/app";
import type { Plugin } from "../engine";
import { AudioPlugin } from "./audio";
import { ComputePlugin } from "./compute";
import { InputPlugin } from "./input";
import { PhysicsPlugin } from "./physics";
import { PlayerPlugin } from "./player";
import { RasterPlugin } from "./raster";
import { RenderPlugin } from "./render";
import { TransformsPlugin } from "./transforms";
import { ViewportPlugin } from "./viewport";
import { shallotDark } from "./loading";

export const DEFAULT_PLUGINS: readonly Plugin[] = [
    TransformsPlugin,
    InputPlugin,
    ComputePlugin,
    ViewportPlugin,
    RenderPlugin,
    RasterPlugin,
    AudioPlugin,
    PhysicsPlugin,
    PlayerPlugin,
];

setDefaultPlugins(DEFAULT_PLUGINS);
setDefaultLoading(shallotDark);
