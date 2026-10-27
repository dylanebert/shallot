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
            name: "Events",
            scenes: [
                { id: "contact", name: "Contact" },
                { id: "hit", name: "Hit" },
                { id: "joint", name: "Joint" },
                { id: "sensor", name: "Sensor Visit" },
            ],
        },
        {
            name: "Joints",
            scenes: [
                { id: "ballAndChain", name: "Ball and Chain" },
                { id: "bridge", name: "Bridge" },
                { id: "distance", name: "Distance Joint" },
                { id: "filter", name: "Filter" },
                { id: "foldOut", name: "Fold Out" },
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
