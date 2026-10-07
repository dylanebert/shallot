import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { run } from "../native";
const [before, after, scene, thread = "1", label = "ab"] = process.argv.slice(2);
const windows: Record<string, [number, number]> = { rain: [280,320], junkyard: [180,200], joint_grid: [20,40] };
if (!before || !after || !windows[scene]) throw new Error("usage: bun diagnostics/box3d-parity/kernel-gap/ab.ts BEFORE_ROOT AFTER_ROOT SCENE THREAD LABEL");
const [from,to] = windows[scene];
const bundle = async (root: string) => {
    const out = join(root,"ab-bundle"); mkdirSync(out,{recursive:true});
    const b = await Bun.build({entrypoints:[join(root,"diagnostics/box3d-parity/scenes.ts")],outdir:out,target:"node",format:"esm"});
    if(!b.success) throw new Error(b.logs.join("\n"));
    return join(out,"scenes.js");
};
const a = await bundle(before), b = await bundle(after);
const hashes = (s: string) => s.split("\n").filter(l => /^\d+ 0x/.test(l)).join("\n");
const reference = readFileSync(join(import.meta.dir,`${scene}-${thread}-native.txt`),"utf8");
const result: object[] = [];
const start = performance.now();
for (const [side,entry] of [["B",b],["A",a],["B",b],["A",a],["B",b],["A",a]]) {
    const text = run(["node",entry,scene,thread,String(to)],{PROFILE:String(from),WALL:String(from)});
    if(hashes(text)!==hashes(reference)) throw new Error(`${side}: hashes differ`);
    result.push({side,rows:text.split("\n").filter(l=>l.startsWith("F ")||l.startsWith("T "))});
}
writeFileSync(join(import.meta.dir,`${label}-${thread}.json`),JSON.stringify(result,null,2)+"\n");
console.log(`${label}/${thread}: B A B A B A, hashes equal; ${(performance.now()-start).toFixed(0)} ms`);
