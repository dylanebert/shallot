import { setFlagsFromString } from "node:v8";
import { CameraBackground } from "@dylanebert/shallot";
import create, { control } from "./turnover.entry";

export { control };

// The oracle's subject with diagnosis knobs, as space-separated words: a V8 flag, set before the subject
// tiers; `undeclared` drops the wrapper declarations; `no-background` drops the camera's sky backdrop.

export default async function probe(input = "") {
    const words = input.split(" ").filter(Boolean);
    for (const word of words) if (word.startsWith("--")) setFlagsFromString(word);
    const subject = await create();
    if (words.includes("no-background")) subject.world.remove(subject.terrain.camera, CameraBackground);
    return words.includes("undeclared") ? { ...subject, allocationExpressions: [], allocationFunctions: [] } : subject;
}
