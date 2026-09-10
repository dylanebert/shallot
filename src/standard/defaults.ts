import type { Plugin } from "../engine";
import { setDefaultLoading, setDefaultPlugins } from "../engine/app";
import { GlazePlugin } from "./glaze";
import { InputPlugin } from "./input";
import { shallotDark } from "./loading";
import { PartPlugin } from "./part";
import { RenderPlugin } from "./render";
import { SearPlugin } from "./sear";
import { SlabPlugin } from "./slab";
import { TransformsPlugin } from "./transforms";

export const DEFAULT_PLUGINS: readonly Plugin[] = [
    SlabPlugin,
    TransformsPlugin,
    InputPlugin,
    RenderPlugin,
    PartPlugin,
    SearPlugin,
    GlazePlugin,
];

setDefaultPlugins(DEFAULT_PLUGINS);
setDefaultLoading(shallotDark);
