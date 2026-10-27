import { expect, test } from "bun:test";
import { formatValue } from "./control-format";
import { type Controls, controlValues, enabledControls, type Scene, scenes } from "./scenes";

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
