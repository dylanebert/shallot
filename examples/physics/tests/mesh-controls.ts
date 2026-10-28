export function meshControl(name: string, value: number): { name: string; value: number } {
    const shape = ["Sphere", "Capsule", "Box", "Cylinder"].indexOf(name);
    const sign = /^(Neg|Pos) ([XYZ])$/.exec(name);
    if (shape >= 0) return { name: "Shape", value: shape };
    if (sign) return { name: `Scale ${sign[2]}`, value: sign[1] === "Neg" ? -1 : 1 };
    if (name === "median split" || name === "sah binning")
        return { name: "Split", value: Number(name === "median split") };
    return { name, value };
}
