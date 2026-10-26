import { type Plugin, StandardPhysicsPlugin } from "@dylanebert/shallot";
import {
    DiagnosticsOverlayPlugin,
    LinesPlugin,
    OrbitPlugin,
    TextPlugin,
} from "@dylanebert/shallot/extras";
import { BenchmarkSamplePlugin } from "./benchmarks";
import { EventSamplePlugin, SampleInputPlugin } from "./events";
import { GrabPlugin } from "./grab";
import { createViewer } from "./view";

export const Viewer = { create: createViewer };

export const AppPlugin: Plugin = {
    name: "PhysicsApp",
    dependencies: [
        StandardPhysicsPlugin,
        EventSamplePlugin,
        BenchmarkSamplePlugin,
        SampleInputPlugin,
        TextPlugin,
        OrbitPlugin,
        LinesPlugin,
        GrabPlugin,
        {
            name: "SampleViewer",
            dependencies: [OrbitPlugin],
            initialize(world) {
                world.resource(Viewer);
            },
        },
        DiagnosticsOverlayPlugin,
    ],
};
