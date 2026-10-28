import { expect, test } from "bun:test";
import { categories, scenes } from "./scenes";

test("each picker label selects its scene in its category, exactly once", () => {
    expect(categories).toEqual([
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
            name: "Continuous",
            scenes: [
                { id: "bounceHouse", name: "Bounce House" },
                { id: "humpMesh", name: "Hump Mesh" },
                { id: "isFast", name: "Is Fast" },
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
    ]);
    expect(
        categories.flatMap((category) => category.scenes.map((scene): string => scene.id)).sort(),
    ).toEqual(Object.keys(scenes).sort());
});

test("each category lists its scenes alphabetically, as Box3D's samples app sorts them", () => {
    for (const category of categories) {
        const names = category.scenes.map((scene) => scene.name);
        expect(names, category.name).toEqual([...names].sort((a, b) => a.localeCompare(b)));
    }
});
