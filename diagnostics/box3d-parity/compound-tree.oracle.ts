import { expect, setDefaultTimeout, test } from "bun:test";
import gold from "../../src/standard/physics/shapes/geometry.gold.json";
import cases from "./compound-tree.inputs.json";
import { nativeSseOutput } from "./native-evidence";

setDefaultTimeout(0);
for (const fixture of cases) {
    test(`compound inner tree ${fixture.name} equals the pinned native rebuild`, () => {
        const input = `${fixture.boxes.length}\n${fixture.boxes.map(box => box.join(" ")).join("\n")}\n`;
        const tree = JSON.parse(nativeSseOutput("compound-tree.c", input));
        expect(gold.compounds.find(c => c.name === fixture.name)!.tree).toEqual(tree);
    });
}
