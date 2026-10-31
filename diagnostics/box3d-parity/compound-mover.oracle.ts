import {expect,test} from "bun:test";
import gold from "../../src/standard/physics/collision/mover.gold.json";
import {nativeSseOutput} from "./native-evidence";
export function compoundMoverInput():string{return [gold.compound.length,...gold.compound.map(g=>[...g.mover.center1,...g.mover.center2,g.mover.radius].join(" "))].join("\n")+"\n";}
test("the frozen compound mover vectors retain native plane order",()=>{
    expect(gold.compound.map(g=>({count:g.count,planes:g.planes}))).toEqual(JSON.parse(nativeSseOutput("compound-mover.c",compoundMoverInput())));
});
