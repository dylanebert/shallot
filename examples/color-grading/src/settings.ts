import { ColorGrading, type World } from "@dylanebert/shallot";
import { Scene } from "./scene";

type Global = "exposure" | "temperature" | "tint" | "hue";
type Section = "saturation" | "contrast" | "gamma" | "gain" | "lift";
type Control = {
    group: string;
    label: string;
    field: Global | Section;
    axis?: "x" | "y" | "z";
    min: number;
    max: number;
    value: number;
};
export const settings: Control[] = [
    { group: "Global", label: "Exposure", field: "exposure", min: -4, max: 4, value: 0 },
    { group: "Global", label: "Temperature", field: "temperature", min: -0.1, max: 0.1, value: 0 },
    { group: "Global", label: "Tint", field: "tint", min: -0.1, max: 0.1, value: 0 },
    { group: "Global", label: "Hue", field: "hue", min: -3.14, max: 3.14, value: 0 },
];
for (const [group, axis] of [
    ["Highlights", "z"],
    ["Midtones", "y"],
    ["Shadows", "x"],
] as const) {
    for (const field of ["saturation", "contrast", "gamma", "gain", "lift"] as const) {
        settings.push({
            group,
            label: field[0].toUpperCase() + field.slice(1),
            field,
            axis,
            min: field === "lift" ? -0.5 : 0,
            max: field === "lift" ? 0.5 : 2,
            value: field === "lift" ? 0 : 1,
        });
    }
}
export function setSetting(world: World, index: number, value: number): void {
    const control = settings[index];
    const camera = world.resource(Scene).camera;
    const storage = world.storage(ColorGrading);
    if (control.axis) storage[control.field as Section][control.axis].set(camera, value);
    else storage[control.field as Global].set(camera, value);
}
