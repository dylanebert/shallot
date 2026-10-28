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
import { CharacterSamplePlugin } from "./character";
import { CollisionPlugin } from "./collision";
import { CollisionDebugPlugin } from "./collision-debug";
import { ContinuousSamplePlugin } from "./continuous";
import { EventSamplePlugin, SampleInputPlugin } from "./events";
import { GeometricMoverPlugin } from "./geometric-mover";
import { GeometryPlugin } from "./geometry";
import { GrabPlugin } from "./grab";
import { JointSamplePlugin } from "./joints";
import { ManifoldPlugin } from "./manifold";
import { MeshSamplePlugin } from "./mesh";
import { RagdollPlugin } from "./ragdoll";
import { RigidBodyPlugin } from "./rigid-body";
import { ShapesSamplePlugin } from "./shapes";
import { StackingSamplePlugin } from "./stacking";
import { createViewer } from "./view";
import { VillagePlugin } from "./village";

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
        RagdollPlugin,
        CollisionPlugin,
        CollisionDebugPlugin,
        GeometryPlugin,
        ManifoldPlugin,
        MeshSamplePlugin,
        ShapesSamplePlugin,
        JointSamplePlugin,
        BodySamplePlugin,
        CharacterSamplePlugin,
        ContinuousSamplePlugin,
        GeometricMoverPlugin,
        VillagePlugin,
        RigidBodyPlugin,
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
