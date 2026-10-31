import {expect,setDefaultTimeout,test} from "bun:test";
import {BodyType} from "../../src/standard/physics/common/types";
import {createSphereShape} from "../../src/standard/physics/shapes/shape";
import {createTestWorld,createTestBody,destroyTestWorld,stepTestWorld,testShapeDef} from "../../src/standard/physics/testing";
import {worldCounters,worldSetCustomFilterCallback} from "../../src/standard/physics/world/world";
import inputs from "../../src/standard/physics/collision/broad-pairs.inputs.json";
import gold from "../../src/standard/physics/collision/broad-pairs.gold.json";
import {nativeSseOutput} from "./native-evidence";
import {assertPublicOracleKernel} from "./oracle-kernel";
setDefaultTimeout(0);
test("the complete overlapping pair sets equal native, including sets beyond the former per-proxy limit",async()=>{
    await assertPublicOracleKernel();
    const native=nativeSseOutput("broad-pairs.c",inputs.map(row=>row.join(" ")).join("\n")+"\n").trim().split("\n").map(line=>JSON.parse(line));
    expect(native).toEqual(gold);
    for(const g of gold){const world=createTestWorld({gravity:{x:0,y:0,z:0},enableSleep:false});try{
        for(let i=0;i<g.count;i++){const body=createTestBody(world,{type:BodyType.Dynamic});createSphereShape(world,body,testShapeDef({enableCustomFiltering:g.mode===2||(g.mode===1&&i===0)}),{center:{x:0,y:0,z:0},radius:0.5});}
        let calls=0;if(g.mode)worldSetCustomFilterCallback(world,()=>{calls++;return true;});stepTestWorld(world,0);expect(worldCounters(world).contactCount).toBe(g.contacts);expect(calls).toBe(g.calls);
    }finally{destroyTestWorld(world);}}
});
