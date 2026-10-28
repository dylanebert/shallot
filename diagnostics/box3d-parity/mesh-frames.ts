import { createHash } from "node:crypto";
import { Shape, ShapeKind, Transform, type World } from "../../src";
import { heightFieldBuild, meshBuild } from "../../src/standard/physics/geometry-builds";
import { drawHeightField, drawMeshData } from "../../src/standard/physics/shapes/readgeometry";
const digest = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
export function nativeMeshFrames(text: string): string[] {
    const data = new Map<string, {kind:string;data:unknown}>();
    const frames: string[] = [];
    for(const line of text.trimEnd().split("\n")) {
        const r = JSON.parse(line,(_key,value)=>typeof value === "number"?Math.fround(value):value);
        if(r.bytes) {
            const bytes=Buffer.from(r.bytes,"hex");
            const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
            data.set(r.id,{kind:r.kind,data:r.kind==="mesh"?drawMeshData(v,0):drawHeightField(v,0)});
        } else frames.push(digest(r.frame.map(({id,p,q,scale}:{id:string;p:number[];q:number[];scale:number[]})=>({p,q,scale,...data.get(id)!}))));
    }
    return frames;
}
export function meshFrame(world:World):string {
    return digest([...world.query([Shape])].filter(eid=>world.get(eid,Shape)!.kind===ShapeKind.Mesh || world.get(eid,Shape)!.kind===ShapeKind.HeightField).map(eid=>{
        const shape=world.get(eid,Shape)!;
        const pose=world.get(shape.body||eid,Transform);
        const mesh=shape.kind===ShapeKind.Mesh;
        return {p:pose?.translation??[0,0,0],q:pose?.rotation??[0,0,0,1],scale:shape.scale,kind:mesh?"mesh":"height",data:mesh?meshBuild(world,shape.geometry):heightFieldBuild(world,shape.geometry)};
    }));
}
