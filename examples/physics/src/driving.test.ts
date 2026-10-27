import { expect, test } from "bun:test";
import { formatValue } from "./control-format";
import { scenes } from "./scenes";

test("Driving's rows retain DrawControls bounds and decimal formats without ImGui suffixes", () => {
    expect(
        Object.values(scenes.driving.controls).map(({ label, value, min, max, step, decimals }) => [
            label,
            formatValue(value, decimals),
            min,
            max,
            step,
        ]),
    ).toEqual([
        ["Suspension Min", "-0.2", -10, 10, 0.1],
        ["Suspension Max", "0.2", -10, 10, 0.1],
        ["Suspension Hertz", "4.0", 0, 10, 0.1],
        ["Suspension Damping", "0.7", 0, 2, 0.1],
        ["Max Torque", "5", 0, 100, 1],
        ["Speed", "30", 0, 100, 1],
        ["Steering Hertz", "10.0", 0, 10, 0.1],
        ["Steering Damping", "0.7", 0, 2, 0.1],
        ["Steering Torque", "5.0", 0, 20, 0.1],
        ["Steering Min Deg", "-45", -90, 0, 1],
        ["Steering Max Deg", "45", 0, 90, 1],
    ]);
});
