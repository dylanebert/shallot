import { expect, setDefaultTimeout, test } from "bun:test";
import gold from "../../src/standard/physics/collision/tree.gold.json";
import { nativeSseOutput } from "./native-evidence";
import { treeStreamInput, treeStreamObservations } from "./tree-stream";
setDefaultTimeout(0);
test("the tree mutation and query stream's checkpoints and observations equal native at the pin",()=>{
    const actual=nativeSseOutput("tree-stream.c",treeStreamInput(gold)).trim().split("\n").map(line=>JSON.parse(line));
    expect(treeStreamObservations(gold)).toEqual(actual);
});
