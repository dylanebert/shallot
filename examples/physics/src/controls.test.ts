import { expect, test } from "bun:test";
import { formatValue } from "./control-format";
import { type Controls, controlValues, enabledControls, type Scene, scenes } from "./scenes";

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
        test(`${scene} slider endpoints and defaults format without loss in the value allocation`, () => {
            for (const control of sliders) {
                expect(control.min!).toBeLessThan(control.max!);
                for (const value of [control.min!, control.max!, control.value]) {
                    expect(Number.isFinite(value)).toBe(true);
                    expect(Number(formatValue(value, control.decimals))).toBe(value);
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
