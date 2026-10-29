import { expect, test } from "bun:test";
import { formatReadout, formatValue } from "./control-format";
import { manifoldNames, manifoldPointLimits } from "./manifold";
import { type Controls, controlValues, enabledControls, type Scene, scenes } from "./scenes";

test("Determinism retains only read-only sample claims", () => {
    for (const scene of [
        "fallingRagdolls",
        "wavePile",
        "meshDropDeterminism",
        "querySpawn",
    ] as const) {
        const controls = Object.values(scenes[scene].controls);
        expect(controls.map((c) => c.label)).toEqual(
            scene === "querySpawn"
                ? ["Sleep step", "Hash", "Spawned", "Query hits", "Query hash"]
                : ["Sleep step", "Hash"],
        );
        expect(controls.every((c) => !!c.read)).toBe(true);
    }
});

test("World retains offset presets and read-only position rows without build facts", () => {
    const stack = scenes.farStack.controls;
    expect(Object.keys(stack)).toEqual([
        "origin",
        "10km",
        "100km",
        "1000km",
        "10000km",
        "Offset",
        "Height",
    ]);
    for (const name of ["origin", "10km", "100km", "1000km", "10000km"])
        expect((stack as Controls)[name]).toMatchObject({ label: name, button: true });
    expect(stack.Offset).toMatchObject({ label: "Offset, km", decimals: 1 });
    expect(stack.Height).toMatchObject({ label: "Top height, m", decimals: 4 });
    for (const scene of ["farPyramid", "farRagdolls", "farMeshDrop"] as const) {
        const controls = scenes[scene].controls;
        expect(controls.Offset).toMatchObject({ label: "Offset, km", decimals: 0 });
        for (const row of Object.values(controls)) expect(row.read).toBeFunction();
    }
    expect(Object.keys(scenes.farPyramid.controls)).toEqual(["Offset"]);
    expect(Object.keys(scenes.farRagdolls.controls)).toEqual(["Offset"]);
    expect(Object.keys(scenes.farMeshDrop.controls)).toEqual(["Offset", "Failure"]);
});

test("Robustness retains native recovery controls and only the overflow pile's claim readouts", () => {
    expect(scenes.highMassRatio1.controls).toBeUndefined();
    expect(scenes.tinyPyramid.controls).toBeUndefined();
    const controls = scenes.overlapRecovery.controls;
    expect(Object.keys(controls)).toEqual([
        "Extent",
        "Base Count",
        "Overlap",
        "Speed",
        "Hertz",
        "Damping Ratio",
        "Reset Scene",
    ]);
    for (const [name, value, min, max, step, decimals] of [
        ["Extent", 0.5, 0.1, 1, 0.1, 1],
        ["Base Count", 4, 1, 10, 1, 0],
        ["Overlap", 0.25, 0, 1, 0.01, 2],
        ["Speed", 3, 0, 10, 0.1, 1],
        ["Hertz", 30, 0, 240, 1, 0],
        ["Damping Ratio", 10, 0, 20, 0.1, 1],
    ] as const)
        expect(controls[name]).toMatchObject({ label: name, value, min, max, step, decimals });
    expect(controls["Reset Scene"].button).toBe(true);
    const rows = scenes.overflowColorPile.controls;
    expect(Object.keys(rows)).toEqual(["Neighbors", "Overflow contacts", "Total contacts"]);
    for (const row of Object.values(rows)) {
        expect(row.read).toBeFunction();
        expect(row.decimals).toBe(0);
    }
});

test("Manifold retains the cache toggle, feature choices, native separation formats and reserved contact rows", () => {
    for (const [index, scene] of manifoldNames.entries()) {
        const controls = scenes[scene].controls;
        expect(Object.keys(controls)).toEqual([
            "Use cache",
            "feature",
            "count",
            ...Array.from({ length: manifoldPointLimits[index]! }, (_, i) => `separation${i}`),
        ]);
        expect(controls["Use cache"]).toMatchObject({ label: "Use cache", value: 0 });
        expect(controls.feature).toMatchObject({
            select: true,
            value: 0,
            when: ["Use cache"],
            options: [
                { label: "auto", value: 0 },
                { label: "faceA", value: 1 },
                { label: "faceB", value: 2 },
                { label: "edgePair", value: 3 },
            ],
        });
        expect(controls.count).toMatchObject({ label: "Point count", decimals: 0 });
        for (let i = 0; i < manifoldPointLimits[index]!; i++)
            expect(controls[`separation${i}`]).toMatchObject({
                label: `Separation ${i + 1}, ${scene.startsWith("triangle") ? "cm" : "m"}`,
                decimals: scene.startsWith("triangle") ? 2 : 3,
            });
    }
});

test("Geometry retains native bounds, formats, choices, Refresh and result rows", () => {
    expect(scenes.geometryHull.controls).toBeUndefined();
    const box = scenes.boxHull.controls!;
    expect(Object.values(box).filter((c) => !c.read)).toHaveLength(13);
    for (const axis of ["X", "Y", "Z"]) {
        expect(box[`h ${axis}`]).toMatchObject({ min: 0.1, max: 2, step: 0.1, decimals: 1 });
        expect(box[`c ${axis}`]).toMatchObject({ min: -2, max: 2, step: 0.1, decimals: 1 });
        expect(box[`r ${axis}`]).toMatchObject({ min: -180, max: 180, step: 1, decimals: 0 });
        expect(box[`s ${axis}`]).toMatchObject({ min: -2, max: 2, step: 0.1, decimals: 1 });
    }
    expect(box.Refresh!.button).toBe(true);
    const transform = scenes.hullTransform.controls!;
    for (const axis of ["x", "y", "z"]) {
        expect(transform[`s${axis}`]).toMatchObject({
            value: 1,
            min: -2,
            max: 2,
            step: 0.1,
            decimals: 1,
        });
        expect(transform[`r${axis}`]).toMatchObject({
            value: 0,
            min: -180,
            max: 180,
            step: 1,
            decimals: 0,
        });
        expect(transform[`p${axis}`]).toMatchObject({
            value: 0,
            min: -1,
            max: 1,
            step: 0.1,
            decimals: 1,
        });
    }
    expect(scenes.hullReduction.controls!.type!.options).toEqual([
        { label: "Box", value: 0 },
        { label: "Sphere", value: 1 },
    ]);
    expect(scenes.hullReduction.controls!.count).toMatchObject({
        value: 16,
        min: 4,
        max: 128,
        step: 1,
    });
    expect(scenes.capsuleMass.controls!.sides).toMatchObject({ value: 6, min: 3, max: 6, step: 1 });
    for (const [scene, count] of [
        ["hullReduction", 3],
        ["hullTransform", 6],
        ["capsuleMass", 12],
    ] as const)
        expect(Object.values(scenes[scene].controls!).filter((c) => c.read)).toHaveLength(count);
});

test("Collision rows retain native choices, bounds and formats with units in fixed-width labels", () => {
    const cast = scenes.castWorld.controls!;
    expect(Object.keys(cast)).toEqual([
        "Cast Type",
        "Radius",
        "Mode",
        "Initial Overlap",
        "Spheres",
        "Capsules",
        "Hulls",
        "Meshes",
        "Height Field",
        "Destroy Shape",
        "Material 1",
        "Triangle 1",
        "Material 2",
        "Triangle 2",
        "Material 3",
        "Triangle 3",
    ]);
    expect(cast["Cast Type"]!.options!.map((o) => o.label)).toEqual([
        "Ray",
        "Sphere",
        "Capsule",
        "Box",
    ]);
    expect(cast.Mode!.options!.map((o) => o.label)).toEqual([
        "Any",
        "Closest",
        "Multiple",
        "Sorted",
    ]);
    expect(cast.Radius).toMatchObject({ value: 0.5, min: 0.1, max: 2, step: 0.1, decimals: 1 });
    const mesh = scenes.meshScale.controls!;
    expect(Object.keys(mesh)).toEqual([
        "Scale X",
        "Scale Y",
        "Scale Z",
        "Start Y",
        "Start Z",
        "sphere Cast",
    ]);
    for (const name of ["Scale X", "Scale Y", "Scale Z", "Start Y", "Start Z"])
        expect(mesh[name]).toMatchObject({ min: -2, max: 2, step: 0.1, decimals: 1 });
    expect(Object.keys(scenes.shapeCast.controls!)).toEqual(["Initial Overlap"]);
    expect(Object.keys(scenes.initialOverlap.controls!)).toEqual(["initial overlap"]);
    const long = scenes.longRayCast.controls!;
    expect(long["Ray Length"]).toMatchObject({
        label: "Ray Length, km",
        value: 1,
        min: 1,
        max: 10000,
        step: 1,
        decimals: 0,
        logarithmic: true,
    });
    expect(long["Cone Angle"]).toMatchObject({
        label: "Cone Angle, deg",
        value: 5,
        min: 0,
        max: 12,
        step: 0.1,
        decimals: 1,
    });
    expect(formatValue(10000, long["Ray Length"]!.decimals)).toBe("10000");
    expect(formatValue(12, long["Cone Angle"]!.decimals)).toBe("12.0");
    for (const name of ["Sphere", "Capsule", "Hull", "Mesh", "Height Field"]) {
        expect(long[`${name} error`]!.read).toBeFunction();
        expect(long[`${name} failures`]!.read).toBeFunction();
    }
    for (const scene of ["rayCurtain", "overlapWorld", "capsuleCastRay"] as const)
        expect(scenes[scene].controls).toBeUndefined();
});

test("Ragdoll retains only the native tuning rows and Respawn with step-based formats", () => {
    for (const [scene, hertz] of [
        ["ragdollBox", 1],
        ["ragdollMesh", 2],
    ] as const) {
        const controls = scenes[scene].controls;
        expect(Object.keys(controls)).toEqual(["Joint Friction", "Hertz", "Damping", "Respawn"]);
        expect(controls["Joint Friction"]).toMatchObject({
            value: 5,
            min: 0,
            max: 20,
            step: 1,
            decimals: 0,
        });
        expect(controls.Hertz).toMatchObject({
            value: hertz,
            min: 0,
            max: 20,
            step: 0.1,
            decimals: 1,
        });
        expect(controls.Damping).toMatchObject({
            value: 0.7,
            min: 0,
            max: 4,
            step: 0.1,
            decimals: 1,
        });
        expect(controls.Respawn).toMatchObject({ button: true });
    }
    expect(scenes.ragdollPile.controls).toBeUndefined();
    expect(scenes.ragdollIncline.controls).toBeUndefined();
});

test("read-only values keep native precision without the slider's numeric-width fallback", () => {
    expect(formatReadout(34418.23828125, undefined, 6)).toBe("34418.2");
    expect(formatReadout(12.3456, 4)).toBe("12.3456");
    expect(formatReadout(0.1, 4)).toBe("0.1000");
    expect(formatReadout(0.000012345, undefined, 6)).toBe("1.2345e-5");
    expect(formatReadout(0, undefined, 6)).toBe("0");
});

test("Viewer retains native mesh choices and slider formats; inspector and benchmark outputs are read-only", () => {
    const controls = scenes.meshViewer.controls;
    expect(Object.keys(controls)).toEqual([
        "index",
        "Split",
        "concave edges",
        "weld vertices",
        "tolerance",
        "draw level",
        "triangle count",
        "vertex count",
        "degenerate count",
        "node area",
        "height",
    ]);
    expect(controls.index).toMatchObject({ value: 0, min: 0, max: 3, step: 1, decimals: 0 });
    expect(controls.Split).toMatchObject({
        value: 1,
        select: true,
        options: [
            { label: "median split", value: 1 },
            { label: "sah binning", value: 0 },
        ],
    });
    expect(controls["concave edges"]).toMatchObject({ value: 1 });
    expect(controls["weld vertices"]).toMatchObject({ value: 1 });
    expect(controls.tolerance).toMatchObject({
        value: 1.5,
        min: 0,
        max: 10,
        step: 0.1,
        decimals: 1,
    });
    expect(formatValue(1.5, controls.tolerance!.decimals)).toBe("1.5");
    expect(controls["draw level"]).toMatchObject({ value: -1, min: -1, step: 1, decimals: 0 });
    for (const name of [
        "triangle count",
        "vertex count",
        "degenerate count",
        "node area",
        "height",
    ])
        expect(controls[name]!.read).toBeFunction();
    expect(controls["node area"]!.precision).toBe(6);
    expect(Object.keys(scenes.meshCreation.controls)).toEqual([
        "triangle count",
        "total time",
        "time per mesh",
    ]);
    for (const control of Object.values(scenes.meshCreation.controls))
        expect(control.read).toBeFunction();
    expect(scenes.meshCreation.controls["total time"].decimals).toBe(4);
    expect(scenes.meshCreation.controls["time per mesh"].decimals).toBe(4);
});

test("Mesh rows retain the four shape choices and native scale bounds without telemetry", () => {
    expect(scenes.voxel.controls).toBeUndefined();
    expect(scenes.hollowBox.controls).toBeUndefined();
    for (const scene of ["gridMesh", "bigBoxMesh", "boxMesh"] as const) {
        const controls = scenes[scene].controls;
        expect(Object.keys(controls)).toEqual(["Shape", "Scale X", "Scale Z"]);
        expect(controls.Shape!.options).toEqual(
            ["Sphere", "Capsule", "Box", "Cylinder"].map((label, value) => ({ label, value })),
        );
        expect(controls.Shape!.select).toBe(true);
        expect(controls.Shape!.value).toBe(scene === "boxMesh" ? 2 : 3);
        for (const label of ["Scale X", "Scale Z"])
            expect(controls[label]).toMatchObject({
                label,
                min: -2,
                max: 2,
                step: 0.1,
                decimals: 1,
                value: scene === "gridMesh" ? 2 : 1,
            });
    }
});

test("Reflection exposes three sign choices, Neg before Pos, without sample-app telemetry", () => {
    const controls = scenes.meshReflection.controls;
    expect(Object.keys(controls)).toEqual(["Scale X", "Scale Y", "Scale Z"]);
    for (const [axis, value] of [
        ["X", -1],
        ["Y", 1],
        ["Z", 1],
    ] as const) {
        expect(controls[`Scale ${axis}`]).toMatchObject({
            label: `Scale ${axis}`,
            value,
            select: true,
            options: [
                { label: "Neg", value: -1 },
                { label: "Pos", value: 1 },
            ],
        });
    }
});

test("Height Field retains native controls, formats and geometry-dependent ray bounds", () => {
    const controls = scenes.meshHeightField.controls;
    expect(Object.keys(controls)).toEqual([
        "columns",
        "rows",
        "amplitude",
        "holes",
        "ray x",
        "ray z",
        "delta x",
        "delta z",
        "radius",
    ]);
    for (const label of ["columns", "rows"])
        expect(controls[label]).toMatchObject({
            value: 10,
            min: 2,
            max: 500,
            step: 1,
            decimals: 0,
        });
    expect(controls.amplitude).toMatchObject({
        value: 0.75,
        min: 0,
        max: 2,
        step: 0.001,
        decimals: 3,
    });
    expect(controls.radius).toMatchObject({ value: 0.2, min: 0, max: 1, step: 0.001, decimals: 3 });
    expect(controls.holes!.min).toBeUndefined();
    for (const label of ["ray x", "ray z"])
        expect(controls[label]).toMatchObject({ step: 0.1, decimals: 1 });
    for (const label of ["delta x", "delta z"])
        expect(controls[label]).toMatchObject({ step: 1, decimals: 0 });
    expect(controls["ray z"]!.value).toBe(1.01);
    expect(formatValue(controls["ray z"]!.value, controls["ray z"]!.decimals)).toBe("1.0");
    for (const [columns, rows, radius] of [
        [10, 10, 0.2],
        [2, 2, 0],
        [500, 500, 1],
        [12, 15, 0.5],
    ]) {
        const values = {
            ...controlValues("meshHeightField"),
            columns: columns!,
            rows: rows!,
            radius: radius!,
        };
        expect(controls["ray x"]!.bounds!(values)).toEqual([
            -(columns! - 1 + 2 * radius!),
            columns! - 1 + 2 * radius!,
        ]);
        expect(controls["ray z"]!.bounds!(values)).toEqual([
            -(rows! - 1 + 2 * radius!),
            rows! - 1 + 2 * radius!,
        ]);
        expect(controls["delta x"]!.bounds!(values)).toEqual([
            -4 * (columns! - 1),
            4 * (columns! - 1),
        ]);
        expect(controls["delta z"]!.bounds!(values)).toEqual([-4 * (rows! - 1), 4 * (rows! - 1)]);
    }
});

test("Height Field's reachable ray values fit the fixed numeric column", () => {
    const controls = scenes.meshHeightField.controls;
    for (const radius of [0, 0.001, 0.2, 1]) {
        const values = { ...controlValues("meshHeightField"), columns: 500, rows: 500, radius };
        for (const label of ["ray x", "ray z", "delta x", "delta z"]) {
            const control = controls[label]!;
            const [min, max] = control.bounds!(values);
            const steps = Math.ceil((max - min) / control.step!);
            for (let index = 0; index <= steps; index++) {
                const value = Math.min(max, min + index * control.step!);
                const text = formatValue(value, control.decimals);
                expect(text.length, `${label}: ${value}`).toBeLessThanOrEqual(6);
                expect(text).not.toContain("e");
            }
        }
    }
});

test("Village carries neither a Surface row nor sample-app statistics", () => {
    expect(scenes.village.controls).toBeUndefined();
    expect(controlValues("village")).toEqual({});
});

test("Character rows retain Solve, Clip Velocity and two integer read-only overlap counts", () => {
    expect(Object.keys(scenes.geometricMover.controls)).toEqual(["Clip Velocity"]);
    expect(scenes.rigidBody.controls).toBeUndefined();
    expect(scenes.geometricMover.controls["Clip Velocity"]).toMatchObject({
        label: "Clip Velocity",
        value: 1,
    });
    const moverControls: Controls = scenes.geometricMover.controls;
    expect(moverControls["Clip Velocity"]!.min).toBeUndefined();
    expect(moverControls["Clip Velocity"]!.button).toBeUndefined();
    expect(Object.keys(scenes.capsulePlane.controls)).toEqual(["Solve"]);
    expect(scenes.capsulePlane.controls.Solve.button).toBe(true);
    expect(Object.keys(scenes.moverOverlap.controls)).toEqual(["Planes", "Degenerate"]);
    for (const row of Object.values(scenes.moverOverlap.controls)) {
        expect(row.decimals).toBe(0);
        expect(typeof row.read).toBe("function");
    }
});

test("Continuous carries only the samples' labeled controls", () => {
    for (const scene of [
        "thinWall",
        "bounceHouse",
        "spinningStick",
        "needleMesh",
        "humpMesh",
        "isFast",
    ] as const)
        expect(scenes[scene].controls).toBeUndefined();
    expect(Object.keys(scenes.stall.controls)).toEqual(["Launch"]);
    expect(scenes.stall.controls.Launch).toMatchObject({ label: "Launch", value: 0, button: true });
    expect(scenes.bulletVersusStack.controls.Launch).toMatchObject({
        label: "Launch",
        value: 0,
        button: true,
    });
    expect(Object.keys(scenes.meshDrop.controls)).toEqual([
        "Type",
        "Amplitude",
        "Collide",
        "Generate",
        "Auto Generate",
    ]);
    expect(scenes.meshDrop.controls.Type).toMatchObject({
        label: "Type",
        value: 0,
        select: true,
        options: ["box", "capsule", "cylinder", "sphere"].map((label, value) => ({ label, value })),
    });
    expect(scenes.meshDrop.controls.Amplitude).toMatchObject({
        value: 0.5,
        min: 0,
        max: 1,
        step: 0.001,
    });
    expect(scenes.meshDrop.controls.Collide).toMatchObject({ value: 1 });
    for (const name of ["Generate", "Auto Generate"] as const)
        expect(scenes.meshDrop.controls[name]).toMatchObject({ label: name, button: true });
});

test("Sensor Hits retains Bullet and Launch with two integer read-only touch counts", () => {
    const controls: Controls = scenes.sensorHits.controls;
    expect(Object.keys(controls)).toEqual(["Bullet", "Launch", "Begin touch", "End touch"]);
    expect(controls.Bullet).toMatchObject({ label: "Bullet", value: 1 });
    expect(controls.Bullet!.min).toBeUndefined();
    expect(controls.Bullet!.button).toBeUndefined();
    expect(controls.Launch).toMatchObject({ label: "Launch", button: true });
    for (const name of ["Begin touch", "End touch"]) {
        expect(controls[name]).toMatchObject({ value: 0, decimals: 0 });
        expect(controls[name]!.read).toBeFunction();
    }
});

test("Shapes rows retain native choices, slider bounds and stable actions", () => {
    expect(scenes.restitution.controls.Shape.options).toEqual([
        { label: "Sphere", value: 0 },
        { label: "Box", value: 1 },
    ]);
    expect(scenes.wind.controls.Shape.select).toBe(true);
    expect(scenes.wind.controls.Shape.options.map((option) => option.label)).toEqual([
        "Circle",
        "Capsule",
        "Box",
    ]);
    expect(scenes.staticInvoke.controls.Mode.select).toBe(true);
    expect(scenes.staticInvoke.controls.Mode.options.map((option) => option.label)).toEqual([
        "Invoke",
        "Passive",
    ]);
    expect(scenes.staticInvoke.controls.Create.button).toBe(true);
    expect(scenes.staticInvoke.controls.Destroy.button).toBe(true);
    for (const [name, min, max, step, decimals] of [
        ["Wind", -50, 50, 0.1, 1],
        ["Drag", 0, 1, 0.01, 2],
        ["Lift", 0, 4, 0.01, 2],
        ["Count", 1, 60, 1, undefined],
    ] as const)
        expect((scenes.wind.controls as Controls)[name]).toMatchObject({
            min,
            max,
            step,
            ...(decimals === undefined ? {} : { decimals }),
        });
});

test("Bodies rows retain the native choices, bounds and action modes", () => {
    expect(Object.keys(scenes.bodyType.controls)).toEqual(["Type", "Enable"]);
    expect(scenes.bodyType.controls.Type.select).toBe(true);
    expect((scenes.jenga.controls as Controls).Shape!.select).toBeUndefined();
    expect((scenes.fallingTrees.controls as Controls)["Grid size"]!.select).toBeUndefined();
    expect(scenes.bodyType.controls.Type.options.map((option) => option.label)).toEqual([
        "Static",
        "Kinematic",
        "Dynamic",
    ]);
    expect(Object.keys(scenes.weeble.controls)).toEqual(["Teleport", "Explode", "Magnitude"]);
    expect(scenes.weeble.controls.Teleport.button).toBe(true);
    expect(scenes.weeble.controls.Explode.button).toBe(true);
    expect(scenes.weeble.controls.Magnitude).toMatchObject({
        min: -100000,
        max: 100000,
        step: 1,
        decimals: 0,
        value: 20000,
    });
    expect(Object.keys(scenes.disable.controls)).toEqual(["Enable Link", "Enable Ball"]);
    expect(controlValues("disable")).toEqual({ "Enable Link": 1, "Enable Ball": 1 });
});

for (const scene of Object.keys(scenes) as Scene[]) {
    const sliders = Object.values(scenes[scene].controls ?? {}).filter(
        (control) => control.min !== undefined,
    );
    if (sliders.length > 0)
        test(`${scene} slider endpoints and defaults format to their declared precision in the value allocation`, () => {
            for (const control of sliders) {
                expect(control.min!).toBeLessThan(control.max!);
                for (const value of [control.min!, control.max!, control.value]) {
                    expect(Number.isFinite(value)).toBe(true);
                    const displayed = Number(formatValue(value, control.decimals));
                    expect(displayed).toBe(
                        control.decimals === undefined
                            ? value
                            : Number(value.toFixed(control.decimals)),
                    );
                    expect(formatValue(value, control.decimals).length).toBeLessThanOrEqual(6);
                }
            }
        });

    test(`${scene} exposes exactly its controls and gates every dependent`, () => {
        const controls: Controls = scenes[scene].controls ?? {};
        const values = controlValues(scene);
        expect(Object.keys(values)).toEqual(Object.keys(controls));
        for (const [name, control] of Object.entries(controls)) {
            expect(values[name]).toBe(control.value);
            for (const dependency of control.when ?? []) {
                expect(controls[dependency]).toBeDefined();
                expect(controls[dependency]!.min).toBeUndefined();
            }
        }
        const toggles = Object.keys(controls).filter(
            (name) =>
                controls[name]!.min === undefined &&
                !controls[name]!.button &&
                !controls[name]!.read,
        );
        for (let mask = 0; mask < 2 ** toggles.length; mask++) {
            toggles.forEach((name, index) => {
                values[name] = (mask >> index) & 1;
            });
            const remembered = { ...values };
            const enabled = enabledControls(scene, values);
            expect(values).toEqual(remembered);
            for (const [name, control] of Object.entries(controls)) {
                const disabled = (control.when ?? []).some(
                    (dependency) => values[dependency] === 0,
                );
                expect(name in enabled, `${scene}: ${name}, mask ${mask}`).toBe(!disabled);
            }
        }
        expect(enabledControls("boxStack", values)).toEqual({});
    });
}
