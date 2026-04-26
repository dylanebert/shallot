export const Shape = {
    Box: 0,
    Sphere: 1,
    Capsule: 2,
    Plane: 3,
    Mesh: 255,
} as const;

export function shapeToPrimitive(shape: number): number {
    if (shape === Shape.Mesh) return 7;
    return shape & 7;
}
