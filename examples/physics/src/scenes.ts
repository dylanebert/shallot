import {
    Body,
    BodyType,
    FilterJoint,
    Hulls,
    ParallelJoint,
    PrismaticJoint,
    RevoluteJoint,
    Shape,
    SphericalJoint,
    Transform,
    WheelJoint,
    type World,
} from "@dylanebert/shallot";

import { PhysicsWorldDefinition, setAwake } from "@dylanebert/shallot/standard/physics";
import { benchmarkScenes } from "./benchmarks";
import { BodySampleState, bodyScenes } from "./bodies";
import { CastState } from "./cast";
import { CharacterSample, characterScenes } from "./character";
import { CollisionState, collisionScenes } from "./collision";
import { CollisionDebug, debugScenes } from "./collision-debug";
import { compoundScenes } from "./compound";
import { continuousScenes } from "./continuous";
import { eventScenes } from "./events";
import { GeometricMoverState, geometricMoverScene } from "./geometric-mover";
import { Geometry, geometryScenes } from "./geometry";
import { jointScenes } from "./joints";
import { ManifoldSample, manifoldScenes } from "./manifold";
import { meshScenes } from "./mesh";
import { HeightFieldState } from "./mesh-height-field";
import { MeshCreationState, MeshInspectorState } from "./mesh-inspector";
import { PrecessionState } from "./precession";
import { RagdollState, ragdollScenes } from "./ragdoll";
import { RigidBodyState, rigidBodyScene } from "./rigid-body";
import { RobustnessState, robustnessScenes } from "./robustness";
import { MotorizedJoint } from "./roles";
import { ShapeControlsState, shapeScenes, WindState } from "./shapes";
import { StackingImpulses, stackingScenes } from "./stacking";
import { VillageState, villageScene } from "./village";

export type Scene =
    | "geometricMover"
    | "rigidBody"
    | "boxStack"
    | "bridge"
    | "revolute"
    | "prismatic"
    | "wheel"
    | "filter"
    | "parallel"
    | "village"
    | keyof typeof collisionScenes
    | keyof typeof debugScenes
    | keyof typeof manifoldScenes
    | keyof typeof geometryScenes
    | keyof typeof ragdollScenes
    | keyof typeof meshScenes
    | keyof typeof compoundScenes
    | keyof typeof continuousScenes
    | keyof typeof eventScenes
    | keyof typeof benchmarkScenes
    | keyof typeof stackingScenes
    | keyof typeof jointScenes
    | keyof typeof bodyScenes
    | keyof typeof shapeScenes
    | keyof typeof characterScenes
    | keyof typeof robustnessScenes;
export const categories: { name: string; scenes: { id: Scene; name: string }[] }[] = [
    {
        name: "Benchmark",
        scenes: [
            { id: "candyCups", name: "Candy Cups" },
            { id: "chains", name: "Chains" },
            { id: "convexPile", name: "Convex Pile" },
            { id: "destruction", name: "Destruction" },
            { id: "explosion", name: "Explosion" },
            { id: "fallingBoxes", name: "Falling Boxes" },
            { id: "fallingTrees", name: "Falling Trees" },
            { id: "heightField", name: "Height Field" },
            { id: "hull", name: "Hull" },
            { id: "jointGrid", name: "Joint Grid" },
            { id: "junkyard", name: "Junkyard" },
            { id: "largePyramid", name: "Large Pyramid" },
            { id: "largeWorld", name: "Large World" },
            { id: "manyPyramids", name: "Many Pyramids" },
            { id: "rain", name: "Rain" },
            { id: "benchmarkSensor", name: "Sensor" },
            { id: "washer", name: "Washer" },
            { id: "widePyramid", name: "Wide Pyramid" },
        ],
    },
    {
        name: "Bodies",
        scenes: [
            { id: "bodyType", name: "Body Type" },
            { id: "cast", name: "Cast" },
            { id: "classRing", name: "Class Ring" },
            { id: "disable", name: "Disable" },
            { id: "fixedRotation", name: "Fixed Rotation" },
            { id: "gyroscopicPrecession", name: "Gyroscopic Precession" },
            { id: "gyroscopicTorque", name: "Gyroscopic Torque" },
            { id: "kinematic", name: "Kinematic" },
            { id: "lockMixing", name: "Lock Mixing" },
            { id: "offsetKinematic", name: "Offset Kinematic" },
            { id: "spinningBook", name: "Spinning Book" },
            { id: "weeble", name: "Weeble" },
        ],
    },
    {
        name: "Character",
        scenes: [
            { id: "capsulePlane", name: "CapsulePlane" },
            { id: "geometricMover", name: "Geometric Mover" },
            { id: "moverOverlap", name: "MoverOverlap" },
            { id: "rigidBody", name: "Rigid Body" },
        ],
    },
    {
        name: "Collision",
        scenes: [
            { id: "capsuleCastRay", name: "Capsule Cast Ray" },
            { id: "castWorld", name: "Cast World" },
            { id: "distanceDebug", name: "Distance Debug" },
            { id: "initialOverlap", name: "Initial Overlap" },
            { id: "longRayCast", name: "Long Ray Cast" },
            { id: "meshScale", name: "Mesh Scale" },
            { id: "overlapWorld", name: "Overlap World" },
            { id: "rayCurtain", name: "Ray Curtain" },
            { id: "shapeCast", name: "Shape Cast" },
            { id: "shapeCastDebug", name: "Shape Cast Debug" },
            { id: "shapeDistance", name: "Shape Distance" },
            { id: "timeOfImpact", name: "Time of Impact" },
        ],
    },
    {
        name: "Compound",
        scenes: [
            { id: "compoundHulls", name: "Hulls" },
            { id: "meshTile", name: "Mesh Tile" },
            { id: "simpleCompound", name: "Simple" },
            { id: "compoundSpheres", name: "Spheres" },
            { id: "tileFloor", name: "Tile Floor" },
            { id: "village", name: "Village" },
        ],
    },
    {
        name: "Continuous",
        scenes: [
            { id: "bounceHouse", name: "Bounce House" },
            { id: "bulletVersusStack", name: "Bullet vs Stack" },
            { id: "humpMesh", name: "Hump Mesh" },
            { id: "isFast", name: "Is Fast" },
            { id: "meshDrop", name: "Mesh Drop" },
            { id: "needleMesh", name: "Needle Mesh" },
            { id: "spinningStick", name: "Spinning Stick" },
            { id: "stall", name: "Stall" },
            { id: "thinWall", name: "Thin Wall" },
        ],
    },
    {
        name: "Events",
        scenes: [
            { id: "contact", name: "Contact" },
            { id: "hit", name: "Hit" },
            { id: "joint", name: "Joint" },
            { id: "move", name: "Move" },
            { id: "persistentContact", name: "Persistent Contact" },
            { id: "sensorHits", name: "Sensor Hits" },
            { id: "sensor", name: "Sensor Visit" },
        ],
    },
    {
        name: "Manifold",
        scenes: [
            { id: "capsuleVsCapsule", name: "Capsule vs Capsule" },
            { id: "capsuleVsHull", name: "Capsule vs Hull" },
            { id: "capsuleVsSphere", name: "Capsule vs Sphere" },
            { id: "hullVsHull", name: "Hull vs Hull" },
            { id: "hullVsSphere", name: "Hull vs Sphere" },
            { id: "sphereVsSphere", name: "Sphere vs Sphere" },
            { id: "triangleVsCapsule", name: "Triangle vs Capsule" },
            { id: "triangleVsHull", name: "Triangle vs Hull" },
            { id: "triangleVsSphere", name: "Triangle vs Sphere" },
        ],
    },
    {
        name: "Geometry",
        scenes: [
            { id: "boxHull", name: "Box Hull" },
            { id: "capsuleMass", name: "Capsule Mass" },
            { id: "geometryHull", name: "Hull" },
            { id: "hullReduction", name: "Hull Reduction" },
            { id: "hullTransform", name: "Hull Transform" },
        ],
    },
    {
        name: "Joints",
        scenes: [
            { id: "ballAndChain", name: "Ball and Chain" },
            { id: "bridge", name: "Bridge" },
            { id: "distance", name: "Distance Joint" },
            { id: "door", name: "Door" },
            { id: "driving", name: "Driving" },
            { id: "filter", name: "Filter" },
            { id: "foldOut", name: "Fold Out" },
            { id: "gearLift", name: "Gear Lift" },
            { id: "motionLocks", name: "Motion Locks" },
            { id: "motor", name: "Motor Joint" },
            { id: "parallel", name: "Parallel Spring" },
            { id: "prismatic", name: "Prismatic" },
            { id: "revolute", name: "Revolute" },
            { id: "spherical", name: "Spherical" },
            { id: "topDownFriction", name: "Top Down Friction" },
            { id: "weld", name: "Weld" },
            { id: "wheel", name: "Wheel" },
        ],
    },
    {
        name: "Mesh",
        scenes: [
            { id: "bigBoxMesh", name: "Big Box" },
            { id: "boxMesh", name: "Box" },
            { id: "meshCreation", name: "Creation Benchmark" },
            { id: "gridMesh", name: "Grid" },
            { id: "meshHeightField", name: "Height Field" },
            { id: "hollowBox", name: "Hollow Box" },
            { id: "meshReflection", name: "Reflection" },
            { id: "meshViewer", name: "Viewer" },
            { id: "voxel", name: "Voxel" },
        ],
    },
    {
        name: "Ragdoll",
        scenes: [
            { id: "ragdollBox", name: "Box" },
            { id: "ragdollIncline", name: "Incline" },
            { id: "ragdollMesh", name: "Mesh" },
            { id: "ragdollPile", name: "Pile" },
        ],
    },
    {
        name: "Robustness",
        scenes: [
            { id: "highMassRatio1", name: "HighMassRatio1" },
            { id: "overflowColorPile", name: "Overflow Color Pile" },
            { id: "overlapRecovery", name: "Overlap Recovery" },
            { id: "tinyPyramid", name: "Tiny Pyramid" },
        ],
    },
    {
        name: "Shapes",
        scenes: [
            { id: "conveyorBelt", name: "Conveyor Belt" },
            { id: "conveyorMesh", name: "Conveyor Mesh" },
            { id: "highResistance", name: "High Resistance" },
            { id: "inclinedPlane", name: "Inclined Plane" },
            { id: "isotropicFriction", name: "Isotropic Friction" },
            { id: "restitution", name: "Restitution" },
            { id: "rollingResistance", name: "Rolling Resistance" },
            { id: "slideTwist", name: "Slide Twist" },
            { id: "staticInvoke", name: "Static Invoke" },
            { id: "wind", name: "Wind" },
            { id: "windDrop", name: "Wind Drop" },
            { id: "windFlap", name: "Wind Flap" },
        ],
    },
    {
        name: "Stacking",
        scenes: [
            { id: "arch", name: "Arch" },
            { id: "boxStack", name: "Box Stack" },
            { id: "capsuleStack", name: "Capsule Stack" },
            { id: "cardHouse", name: "Card House" },
            { id: "cylinder", name: "Cylinder" },
            { id: "cylinderStack", name: "Cylinder Stack" },
            { id: "dominoes", name: "Dominoes" },
            { id: "doubleDomino", name: "Double Domino" },
            { id: "edgeCrossing", name: "Edge Crossing" },
            { id: "jenga", name: "Jenga Stack" },
            { id: "pyramid2D", name: "Pyramid2D" },
            { id: "singleBox", name: "Single Box" },
            { id: "sphereStack", name: "Sphere Stack" },
            { id: "wedge", name: "Wedge" },
        ],
    },
];
export interface Control {
    label: string;
    value: number;
    min?: number;
    max?: number;
    bounds?: (values: Record<string, number>, world?: World) => [number, number];
    step?: number;
    decimals?: number;
    precision?: number;
    logarithmic?: boolean;
    button?: boolean;
    read?: (world: World) => number | string | null;
    enabled?: (world: World) => boolean;
    options?: readonly { label: string; value: number }[];
    select?: boolean;
    when?: readonly string[];
    set(
        world: World,
        bodies: readonly number[],
        value: number,
        values: Record<string, number>,
    ): void;
}
export type Controls = Record<string, Control>;
function control(
    label: string,
    value: number,
    set: Control["set"],
    min?: number,
    max?: number,
    step?: number,
    when?: string | readonly string[],
): Control {
    return { label, value, set, min, max, step, when: typeof when === "string" ? [when] : when };
}
export function controlValues(scene: Scene): Record<string, number> {
    return Object.fromEntries(
        Object.entries(scenes[scene].controls ?? {}).map(([name, control]) => [
            name,
            control.value,
        ]),
    );
}
export function enabledControls(
    scene: Scene,
    values: Record<string, number>,
    world?: World,
): Controls {
    const controls: Controls | undefined = scenes[scene].controls;
    return Object.fromEntries(
        Object.entries(controls ?? {}).filter(
            ([, control]) =>
                (!control.when || control.when.every((name) => values[name])) &&
                (!world || !control.enabled || control.enabled(world)),
        ),
    );
}
export function changeControl(
    world: World,
    scene: Scene,
    bodies: readonly number[],
    values: Record<string, number>,
    name: string,
    value: number,
): void {
    const controls: Controls | undefined = scenes[scene].controls;
    const selected = controls?.[name];
    if (!selected || selected.read) throw new Error(`Unknown ${scene} control: ${name}`);
    if (!(name in enabledControls(scene, values, world))) return;
    values[name] = value;
    selected.set(world, bodies, value, values);
}
export const scenes = {
    geometricMover: geometricMoverScene,
    rigidBody: rigidBodyScene,
    ...collisionScenes,
    ...debugScenes,
    ...geometryScenes,
    ...manifoldScenes,
    ...ragdollScenes,
    ...shapeScenes,
    ...compoundScenes,
    ...meshScenes,
    village: villageScene,
    ...continuousScenes,
    ...bodyScenes,
    ...eventScenes,
    ...benchmarkScenes,
    ...stackingScenes,
    ...jointScenes,
    ...characterScenes,
    ...robustnessScenes,
    boxStack: {
        construct: boxStack,
        gridExtent: 40,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 10, distance: 46, target: [0, 23, 0] },
        controls: undefined,
    },
    bridge: {
        construct: bridge,
        gridExtent: 25,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 30, distance: 29, target: [-1.25, 17, 0] },
        controls: {
            "Gravity scale": control(
                "Gravity scale",
                1,
                (world, bodies, value) => setGravityScale(world, bodies, value),
                -1,
                1,
                0.1,
            ),
        },
    },
    revolute: {
        construct: revolute,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: {
            Limit: control("Limit", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([RevoluteJoint]);
                if (!eid) return;
                const joint = world.storage(RevoluteJoint);
                joint.enableLimit.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Lower Angle": control(
                "Lower Angle",
                -35,
                (world, bodies, value, values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    values["Lower Angle"] = value = Math.min(value, values["Upper Angle"]!);
                    joint.lowerAngle.set(
                        eid,
                        Math.fround(Math.fround(Math.fround(Math.PI) / 180) * value),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                -180,
                180,
                1,
                "Limit",
            ),
            "Upper Angle": control(
                "Upper Angle",
                35,
                (world, bodies, value, values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    values["Upper Angle"] = value = Math.max(value, values["Lower Angle"]!);
                    joint.upperAngle.set(
                        eid,
                        Math.fround(Math.fround(Math.fround(Math.PI) / 180) * value),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                -180,
                180,
                1,
                "Limit",
            ),
            Motor: control("Motor", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([RevoluteJoint]);
                if (!eid) return;
                const joint = world.storage(RevoluteJoint);
                joint.enableMotor.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Max Torque": control(
                "Max Torque",
                5000,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    joint.maxMotorTorque.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                50000,
                1,
                "Motor",
            ),
            Speed: control(
                "Speed",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    joint.motorSpeed.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                1,
                "Motor",
            ),
            Spring: control("Spring", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([RevoluteJoint]);
                if (!eid) return;
                const joint = world.storage(RevoluteJoint);
                joint.enableSpring.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            Hertz: control(
                "Hertz",
                2,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    joint.hertz.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                10,
                0.1,
                "Spring",
            ),
            Damping: control(
                "Damping",
                0.7,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    joint.dampingRatio.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                2,
                0.1,
                "Spring",
            ),
            Rotation: control(
                "Rotation",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    joint.targetAngle.set(
                        eid,
                        Math.fround(Math.fround(Math.fround(Math.PI) / 180) * value),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                -180,
                180,
                1,
                "Spring",
            ),
        },
    },
    prismatic: {
        construct: prismatic,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: {
            Limit: control("Limit", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([PrismaticJoint]);
                if (!eid) return;
                const joint = world.storage(PrismaticJoint);
                joint.enableLimit.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Lower Translation": control(
                "Lower Translation",
                -1,
                (world, bodies, value, values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    values["Lower Translation"] = value = Math.min(
                        value,
                        values["Upper Translation"]!,
                    );
                    joint.lowerTranslation.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                0.1,
                "Limit",
            ),
            "Upper Translation": control(
                "Upper Translation",
                1,
                (world, bodies, value, values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    values["Upper Translation"] = value = Math.max(
                        value,
                        values["Lower Translation"]!,
                    );
                    joint.upperTranslation.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                0.1,
                "Limit",
            ),
            Motor: control("Motor", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([PrismaticJoint]);
                if (!eid) return;
                const joint = world.storage(PrismaticJoint);
                joint.enableMotor.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Max Force": control(
                "Max Force",
                20,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    joint.maxMotorForce.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                100000,
                1,
                "Motor",
            ),
            Speed: control(
                "Speed",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    joint.motorSpeed.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                1,
                "Motor",
            ),
            Spring: control("Spring", 1, (world, bodies, value, _values) => {
                const [eid] = world.query([PrismaticJoint]);
                if (!eid) return;
                const joint = world.storage(PrismaticJoint);
                joint.enableSpring.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            Hertz: control(
                "Hertz",
                2,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    joint.hertz.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                10,
                0.1,
                "Spring",
            ),
            Damping: control(
                "Damping",
                0.7,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    joint.dampingRatio.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                2,
                0.1,
                "Spring",
            ),
            Translation: control(
                "Translation",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    joint.targetTranslation.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -20,
                20,
                0.1,
                "Spring",
            ),
        },
    },
    wheel: {
        construct: wheel,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 25, pitch: 20, distance: 7, target: [0, 2, 0] },
        controls: {
            "Suspension Limit": control("Suspension Limit", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([WheelJoint]);
                if (!eid) return;
                const joint = world.storage(WheelJoint);
                joint.enableSuspensionLimit.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Min##Limit": control(
                "Min suspension",
                -1,
                (world, bodies, value, values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    values["Min##Limit"] = value = Math.min(value, values["Max##Limit"]!);
                    joint.lowerSuspensionLimit.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                0.1,
                "Suspension Limit",
            ),
            "Max##Limit": control(
                "Max suspension",
                1,
                (world, bodies, value, values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    values["Max##Limit"] = value = Math.max(value, values["Min##Limit"]!);
                    joint.upperSuspensionLimit.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                0.1,
                "Suspension Limit",
            ),
            Motor: control("Motor", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([WheelJoint]);
                if (!eid) return;
                const joint = world.storage(WheelJoint);
                joint.enableSpinMotor.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Max Torque": control(
                "Max Torque",
                20,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.maxSpinTorque.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                100,
                1,
                "Motor",
            ),
            Speed: control(
                "Speed",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.spinSpeed.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                1,
                "Motor",
            ),
            "Suspension Spring": control(
                "Suspension Spring",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.enableSuspensionSpring.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
            ),
            "Hertz##Suspension": control(
                "Hertz suspension",
                2,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.suspensionHertz.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                10,
                0.1,
                "Suspension Spring",
            ),
            "Damping##Suspension": control(
                "Damping suspension",
                0.7,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.suspensionDampingRatio.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                2,
                0.1,
                "Suspension Spring",
            ),
            Steering: control("Steering", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([WheelJoint]);
                if (!eid) return;
                const joint = world.storage(WheelJoint);
                joint.enableSteering.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Hertz##Steering": control(
                "Hertz steering",
                1,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.steeringHertz.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                10,
                0.1,
                "Steering",
            ),
            "Damping##Steering": control(
                "Damping steering",
                0.7,
                (world, bodies, _value, values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint); // Box3D's steering damping control reapplies suspension damping, not steering damping.
                    joint.suspensionDampingRatio.set(eid, values["Damping##Suspension"]!);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                2,
                0.1,
                "Steering",
            ),
            "Degrees##Steering": control(
                "Degrees steering",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.targetSteeringAngle.set(
                        eid,
                        Math.fround(Math.fround(value * Math.fround(Math.PI)) / 180),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                -90,
                90,
                1,
                "Steering",
            ),
            "Steering Limit": control(
                "Steering Limit",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.enableSteeringLimit.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                undefined,
                undefined,
                undefined,
                "Steering",
            ),
            "Min Degrees": control(
                "Min Degrees",
                -45,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.lowerSteeringLimit.set(
                        eid,
                        Math.fround(Math.fround(Math.fround(Math.PI) / 180) * value),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                -90,
                0,
                1,
                ["Steering", "Steering Limit"],
            ),
            "Max Degrees": control(
                "Max Degrees",
                45,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.upperSteeringLimit.set(
                        eid,
                        Math.fround(Math.fround(Math.fround(Math.PI) / 180) * value),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                90,
                1,
                ["Steering", "Steering Limit"],
            ),
        },
    },
    parallel: {
        construct: parallel,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: {
            Hertz: control(
                "Hertz",
                10,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([ParallelJoint]);
                    if (!eid) return;
                    const joint = world.storage(ParallelJoint);
                    joint.hertz.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                5,
                0.1,
            ),
            Damping: control(
                "Damping",
                0.7,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([ParallelJoint]);
                    if (!eid) return;
                    const joint = world.storage(ParallelJoint);
                    joint.dampingRatio.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                2,
                0.1,
            ),
        },
    },
    filter: {
        construct: filter,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: undefined,
    },
} as const;

export function createScene(world: World, scene: Scene): number[] {
    Object.assign(world.resource(RobustnessState), RobustnessState.create());
    Object.assign(world.resource(HeightFieldState), HeightFieldState.create());
    Object.assign(world.resource(MeshInspectorState), MeshInspectorState.create());
    Object.assign(world.resource(MeshCreationState), MeshCreationState.create());
    Object.assign(world.resource(ShapeControlsState), ShapeControlsState.create());
    Object.assign(world.resource(WindState), { shapes: [], joints: [], lift: 0, time: 0 });
    world.resource(StackingImpulses).length = 0;
    Object.assign(world.resource(RagdollState), RagdollState.create());
    Object.assign(world.resource(BodySampleState), BodySampleState.create());
    Object.assign(world.resource(PrecessionState), PrecessionState.create());
    Object.assign(world.resource(CastState), CastState.create());
    Object.assign(world.resource(CollisionState), CollisionState.create());
    Object.assign(world.resource(CollisionDebug), CollisionDebug.create());
    Object.assign(world.resource(Geometry), Geometry.create());
    Object.assign(world.resource(ManifoldSample), ManifoldSample.create());
    Object.assign(world.resource(CharacterSample), CharacterSample.create());
    Object.assign(world.resource(GeometricMoverState), GeometricMoverState.create());
    Object.assign(world.resource(VillageState), VillageState.create());
    Object.assign(world.resource(RigidBodyState), RigidBodyState.create());
    world.resource(PhysicsWorldDefinition).enableAutomaticStep = true;
    world.resource(PhysicsWorldDefinition).workerCount = 1;
    world.resource(PhysicsWorldDefinition).maximumMemory = 4 * 1024 ** 3;
    return scenes[scene].construct(world);
}

function ground(world: World, extent: number): number {
    const geometry = world.resource(Hulls).add({ name: "box", halfExtents: [0.5, 0.5, 0.5] });
    const ground = world.create();
    world.add(ground, Transform, { translation: [0, -1, 0], scale: [2 * extent, 2, 2 * extent] });
    world.add(ground, Body);
    world.add(ground, Shape, { geometry });
    return geometry;
}

function boxStack(world: World): number[] {
    const geometry = ground(world, 40);
    const bodies: number[] = [];
    for (let i = 0; i < 40; i++) {
        const body = world.create();
        world.add(body, Transform, { translation: [0, 0.75 + 1.25 * i, 0] });
        world.add(body, Body, { type: BodyType.Dynamic });
        world.add(body, Shape, { geometry, rollingResistance: 0.1 });
        bodies.push(body);
    }
    return bodies;
}

function bridge(world: World): number[] {
    const geometry = ground(world, 60);
    const bodies: number[] = [];
    const anchor = world.create();
    world.add(anchor, Body);
    let previous = anchor;
    function joint(
        a: number,
        b: number,
        ax: number,
        ay: number,
        bx: number,
        by: number,
        z: number,
    ): void {
        const eid = world.create();
        world.add(eid, SphericalJoint, {
            a,
            b,
            localAnchorA: [ax, ay, z],
            localAnchorB: [bx, by, z],
            constraintHertz: 1000,
            enableSpring: 1,
            hertz: 2,
            dampingRatio: 1,
        });
    }
    for (let i = 0; i < 150; i++) {
        const body = world.create();
        world.add(body, Transform, {
            translation: [-20 + 0.125 * (1 + 2 * i), 20, 0],
            scale: [0.25, 0.25, 1],
        });
        world.add(body, Body, { type: BodyType.Dynamic, linearDamping: 0.1, angularDamping: 0.1 });
        world.add(body, Shape, { geometry, density: 20 });
        for (const z of [-0.5, 0.5])
            joint(previous, body, i === 0 ? -20 : 0.125, i === 0 ? 20 : 0, -0.125, 0, z);
        bodies.push(body);
        previous = body;
    }
    for (const z of [-0.5, 0.5]) joint(previous, anchor, 0.125, 0, 17.5, 20, z);
    return bodies;
}

function hangingBody(world: World, height: number, geometry: number): number {
    const body = world.create();
    world.add(body, Transform, { translation: [0, height, 0] });
    world.add(body, Body, { type: BodyType.Dynamic });
    world.add(body, Shape, { geometry });
    return body;
}

function anchor(world: World): number {
    const body = world.create();
    world.add(body, Transform, { translation: [0, -1, 0] });
    world.add(body, Body);
    return body;
}

function revolute(world: World): number[] {
    ground(world, 20);
    const a = anchor(world);
    const geometry = world.resource(Hulls).add({ name: "pendulum", halfExtents: [0.5, 1.5, 0.25] });
    const b = hangingBody(world, 4, geometry);
    const joint = world.create();
    world.add(joint, MotorizedJoint);
    world.add(joint, RevoluteJoint, {
        a,
        b,
        localAnchorA: [0, 6.5, 0],
        localAnchorB: [0, 1.5, 0],
        drawScale: 2,
        lowerAngle: Math.fround(Math.fround(Math.fround(Math.PI) / 180) * -35),
        upperAngle: Math.fround(Math.fround(Math.fround(Math.PI) / 180) * 35),
        hertz: 2,
        dampingRatio: 0.7,
        maxMotorTorque: 5000,
    });
    return [b];
}

function prismatic(world: World): number[] {
    ground(world, 20);
    const a = anchor(world);
    const geometry = world.resource(Hulls).add({ name: "slider", halfExtents: [0.5, 1.5, 0.25] });
    const b = hangingBody(world, 4, geometry);
    world.storage(Body).gravityScale.set(b, 0);
    const joint = world.create();
    world.add(joint, MotorizedJoint);
    world.add(joint, PrismaticJoint, {
        a,
        b,
        localAnchorA: [0, 6.5, 0],
        localAnchorB: [0, 1.5, 0],
        constraintHertz: 120,
        lowerTranslation: -1,
        upperTranslation: 1,
        enableSpring: 1,
        hertz: 2,
        dampingRatio: 0.7,
        maxMotorForce: 20,
    });
    return [b];
}

const diagonal = 0.7071067690849304;
function wheel(world: World): number[] {
    ground(world, 20);
    const a = anchor(world);
    const geometry = world
        .resource(Hulls)
        .add({ name: "wheel", cylinder: { height: 0.25, radius: 0.4, yOffset: 0, sides: 12 } });
    const b = hangingBody(world, 2, geometry);
    world.storage(Transform).rotation.set(b, diagonal, 0, 0, diagonal);
    const joint = world.create();
    world.add(joint, MotorizedJoint);
    world.add(joint, WheelJoint, {
        a,
        b,
        localAnchorA: [0, 3, 0],
        localRotationA: [0, 0, diagonal, diagonal],
        localRotationB: [-diagonal, 0, 0, diagonal],
        collideConnected: 1,
        enableSuspensionSpring: 0,
        suspensionHertz: 2,
        suspensionDampingRatio: 0.7,
        lowerSuspensionLimit: -1,
        upperSuspensionLimit: 1,
        maxSpinTorque: 20,
        maxSteeringTorque: 20,
        lowerSteeringLimit: Math.fround(Math.fround(Math.fround(Math.PI) / 180) * -45),
        upperSteeringLimit: Math.fround(Math.fround(Math.fround(Math.PI) / 180) * 45),
    });
    return [b];
}

function filter(world: World): number[] {
    const geometry = ground(world, 20);
    const bodies: number[] = [];
    for (const x of [2, -2]) {
        const body = hangingBody(world, 4, geometry);
        world.storage(Transform).translation.set(body, x, 4, 0);
        bodies.push(body);
    }
    world.add(world.create(), FilterJoint, { a: bodies[0]!, b: bodies[1]! });
    return bodies;
}

function parallel(world: World): number[] {
    ground(world, 20);
    const a = anchor(world);
    const walls = [
        { halfExtents: [20, 5, 0.1], position: [0, 5, -20] },
        { halfExtents: [20, 5, 0.1], position: [0, 5, 20] },
        { halfExtents: [0.1, 5, 20], position: [-20, 5, 0] },
        { halfExtents: [0.1, 5, 20], position: [20, 5, 0] },
    ] as const;
    for (const wall of walls) {
        const geometry = world.resource(Hulls).add({
            name: "wall",
            halfExtents: wall.halfExtents,
            transform: { position: wall.position, rotation: [0, 0, 0, 1] },
        });
        world.add(world.create(), Shape, { body: a, geometry });
    }
    const geometry = world
        .resource(Hulls)
        .add({ name: "parallel spring", halfExtents: [0.5, 1.5, 0.25] });
    const b = hangingBody(world, 4, geometry);
    // b3MakeQuatFromAxisAngle(axisX, 0.25f * B3_PI), using Box3D's deterministic trig.
    const x = 0.3837169408798218,
        w = 0.9234507083892822;
    world.storage(Transform).rotation.set(b, x, 0, 0, w);
    const f = Math.fround;
    world.add(world.create(), ParallelJoint, {
        a,
        b,
        localRotationA: [-diagonal, 0, 0, diagonal],
        localRotationB: [
            f(f(w * -diagonal) - f(x * diagonal)),
            0,
            0,
            f(f(w * diagonal) + f(x * -diagonal)),
        ],
        drawScale: 2,
        collideConnected: 1,
        hertz: 10,
        dampingRatio: 0.7,
    });
    return [b];
}

export function setGravityScale(world: World, bodies: readonly number[], scale: number): void {
    const gravity = world.storage(Body).gravityScale;
    for (const body of bodies) gravity.set(body, scale);
}
