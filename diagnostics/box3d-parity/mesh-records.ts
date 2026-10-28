import { createHash } from "node:crypto";
import { PhysicsMeshes, Shape, ShapeKind, Transform, type World } from "../../src";
import { meshBuild } from "../../src/standard/physics/geometry-builds";
import { drawMeshData } from "../../src/standard/physics/shapes/readgeometry";
const digest = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
export function nativeMeshRecords(text: string): string[] {
    return text.trimEnd().split("\n").map(line => {
        const {p,q,scale,bytes:hex} = JSON.parse(line,(_key,value)=>typeof value === "number" ? Math.fround(value) : value);
        const bytes = Buffer.from(hex,"hex");
        return digest({p,q,scale,mesh:drawMeshData(new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),0)});
    });
}
export function meshRecords(world: World): string[] {
    return [...world.query([Shape])].filter(eid=>world.get(eid,Shape)!.kind===ShapeKind.Mesh).map(eid=>{
        const shape = world.get(eid,Shape)!;
        const body = shape.body || eid;
        const pose = world.get(body,Transform);
        if (!world.resource(PhysicsMeshes).get(shape.geometry)) throw new Error("missing mesh");
        return digest({p:pose?.translation ?? [0,0,0],q:pose?.rotation ?? [0,0,0,1],scale:shape.scale,mesh:meshBuild(world,shape.geometry)});
    });
}
