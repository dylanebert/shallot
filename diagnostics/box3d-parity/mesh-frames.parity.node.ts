import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "../../src";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { MeshSamplePlugin } from "../../examples/physics/src/mesh";
import { type Controls, controlValues, createScene, scenes } from "../../examples/physics/src/scenes";
import { meshFrame } from "./mesh-frames";
setDefaultTimeout(TIMEOUT.node);
for(const scene of ["meshReflection","meshHeightField"] as const) for(const controlled of [false,true]) {
    test(`${scene}${controlled?" controls":""} static geometry equals every native frame`,async()=>{
        const base=new URL("../../examples/physics/tests/",import.meta.url);
        const gold=(await Bun.file(new URL(`${scene}${controlled?".controls":""}.geometry.gold.txt`,base)).text()).trim().split("\n");
        const sequence=controlled?(await Bun.file(new URL(`${scene}.controls.txt`,base)).text()).trim().split("\n").map(line=>{const [step,name,value]=line.split("\t");return {step:Number(step),name:name!,value:Number(value)};}):[];
        const app=await createApp({defaults:false,plugins:[MeshSamplePlugin],setup(world){createScene(world,scene);}});
        const values=controlValues(scene);
        try {
            expect(gold).toHaveLength(600);
            for(const [step,expected] of gold.entries()) {
                for(const change of sequence.filter(c=>c.step===step)) {
                    const sign=/^(Neg|Pos) ([XYZ])$/.exec(change.name);
                    const name=sign?`Scale ${sign[2]}`:change.name;
                    const value=sign?(sign[1]==="Neg"?-1:1):change.value;
                    values[name]=value;(scenes[scene].controls as Controls)[name]!.set(app.world,[],value,values);
                }
                app.world.tick();expect(meshFrame(app.world),`step ${step}`).toBe(expected);
            }
        } finally {app.dispose();}
    });
}
