import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { run } from "../native";
const binary = process.argv[2];
const scene = process.argv[3] ?? "junkyard";
const windows: Record<string, [number, number]> = { rain: [280,320], junkyard: [180,200], joint_grid: [20,40] };
if (!binary || !windows[scene]) throw new Error("usage: bun diagnostics/box3d-parity/kernel-gap/times.ts NATIVE_COUNTS rain|junkyard|joint_grid");
const [from,to] = windows[scene];
for (const threads of [1,4]) {
    const text = run([binary, scene, String(threads), String(to)], { TIMERS:"1", PROFILE:String(from) });
    const reference = readFileSync(join(import.meta.dir, `${scene}-${threads}-native.txt`), "utf8");
    const hashes = (s: string) => s.split("\n").filter(l => /^\d+ 0x/.test(l)).join("\n");
    if (hashes(text) !== hashes(reference)) throw new Error(`${scene}/${threads}: hashes differ`);
    writeFileSync(join(import.meta.dir, `${scene}-${threads}-times-native.txt`), text);
    console.log(`${scene}/${threads}: native named timers recorded; hashes equal`);
}
