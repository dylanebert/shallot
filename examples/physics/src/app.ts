import { type Plugin, StandardPhysicsPlugin } from "@dylanebert/shallot";
import {
    DiagnosticsOverlayPlugin,
    LinesPlugin,
    OrbitPlugin,
    TextPlugin,
} from "@dylanebert/shallot/extras";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { BenchmarkSamplePlugin } from "./benchmarks";
import { BodySamplePlugin } from "./bodies";
import { EventSamplePlugin, SampleInputPlugin } from "./events";
import { GrabPlugin } from "./grab";
import { JointSamplePlugin } from "./joints";
import { ShapesSamplePlugin } from "./shapes";
import { StackingSamplePlugin } from "./stacking";
import { createViewer } from "./view";

export const Viewer = { create: createViewer };

export const AppPlugin: Plugin = {
    name: "PhysicsApp",
    initialize(world) {
        world.resource(PhysicsWorldDefinition).maximumMemory = 4 * 1024 ** 3;
    },
    dependencies: [
        StandardPhysicsPlugin,
        EventSamplePlugin,
        BenchmarkSamplePlugin,
        StackingSamplePlugin,
        ShapesSamplePlugin,
        JointSamplePlugin,
        BodySamplePlugin,
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
