import { createHash } from "node:crypto";
import type { CompoundData } from "../../src/standard/physics/shapes/compound";
import type { Vec3, Transform } from "../../src/standard/physics/common/records";
import { drawMeshData } from "../../src/standard/physics/shapes/readgeometry";

const digest = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
const vector = (v: Vec3) => [v.x, v.y, v.z];
const transform = (t: Transform) => ({ p: vector(t.p), q: [t.q.v.x, t.q.v.y, t.q.v.z, t.q.s] });

export function compoundRecord(data: CompoundData): string {
    return digest([
        ...data.capsules.map(({ capsule: c }) => ({ kind: "capsule", center1: vector(c.center1), center2: vector(c.center2), radius: c.radius })),
        ...data.hulls.map(c => ({ kind: "hull", transform: transform(c.transform), points: c.hull.points.map(vector) })),
        ...data.meshes.map(c => ({ kind: "mesh", transform: transform(c.transform), scale: vector(c.scale), mesh: digest(c.meshData) })),
        ...data.spheres.map(({ sphere: s }) => ({ kind: "sphere", center: vector(s.center), radius: s.radius })),
    ]);
}

export function nativeCompoundRecords(text: string): string[] {
    return text.trimEnd().split("\n").map(line => {
        const children = JSON.parse(line, (_key, value) => typeof value === "number" ? Math.fround(value) : value);
        for (const child of children) if (child.kind === "mesh") {
            const bytes = Buffer.from(child.bytes, "hex");
            child.mesh = digest(drawMeshData(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), 0));
            delete child.bytes;
        }
        return digest(children);
    });
}
