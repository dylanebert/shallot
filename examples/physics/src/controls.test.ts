import { expect, test } from "bun:test";
import { type Controls, controlValues, type Scene, scenes, visibleControls } from "./scenes";

for (const scene of Object.keys(scenes) as Scene[]) {
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
            (name) => controls[name]!.min === undefined && !controls[name]!.button,
        );
        for (let mask = 0; mask < 2 ** toggles.length; mask++) {
            toggles.forEach((name, index) => {
                values[name] = (mask >> index) & 1;
            });
            const visible = visibleControls(scene, values);
            for (const [name, control] of Object.entries(controls)) {
                const hidden = (control.when ?? []).some((dependency) => values[dependency] === 0);
                expect(name in visible, `${scene}: ${name}, mask ${mask}`).toBe(!hidden);
            }
        }
        expect(visibleControls("boxStack", values)).toEqual({});
    });
}
