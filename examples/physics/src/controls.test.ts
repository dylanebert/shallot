import { expect, test } from "bun:test";
import { formatValue } from "./control-format";
import { type Controls, controlValues, enabledControls, type Scene, scenes } from "./scenes";

for (const scene of Object.keys(scenes) as Scene[]) {
    test(`${scene} exposes exactly its controls and gates every dependent`, () => {
        const controls: Controls = scenes[scene].controls ?? {};
        const values = controlValues(scene);
        expect(Object.keys(values)).toEqual(Object.keys(controls));
        for (const [name, control] of Object.entries(controls)) {
            expect(values[name]).toBe(control.value);
            if (control.min !== undefined) {
                for (const value of [control.min, control.max!, control.value]) {
                    expect(Number(formatValue(value))).toBe(value);
                    expect(formatValue(value).length).toBeLessThanOrEqual(6);
                }
            }
            for (const dependency of control.when ?? []) {
                expect(controls[dependency]).toBeDefined();
                expect(controls[dependency]!.min).toBeUndefined();
            }
        }
        const toggles = Object.keys(controls).filter(
            (name) => controls[name]!.min === undefined && !controls[name]!.button,
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
