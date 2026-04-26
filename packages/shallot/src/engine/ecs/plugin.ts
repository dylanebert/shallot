import type { State } from "./state";
import type { System } from "./scheduler";
import type { Component } from "./component";
import type { Relation } from "./relation";

/** bundle of components, systems, and lifecycle hooks */
export interface Plugin {
    readonly name: string;
    readonly systems?: readonly System[];
    readonly components?: Record<string, Component>;
    readonly relations?: readonly Relation[];
    readonly dependencies?: readonly Plugin[];
    readonly initialize?: (
        state: State,
        onProgress?: (progress: number) => void,
    ) => void | Promise<void>;
    readonly warm?: (state: State, onProgress?: (progress: number) => void) => void | Promise<void>;
}
