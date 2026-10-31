import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIMEOUT } from "../../src/project/testing.js";
import { nativeBinary, run } from "./native";

setDefaultTimeout(TIMEOUT.node);
const path = "src/standard/physics/solver/joint.gold.json";
const families = ["weld", "revolute", "spherical", "motor"] as const;
const f = (bits: string) => `F(${bits})`;
const assign = (name: string, value: string | number | string[]) => Array.isArray(value)
    ? value.map((v, i) => `((float*)&${name})[${i}]=${f(v)};`).join("")
    : `${name}=${typeof value === "number" ? value : f(value)};`;

/** Capture the frozen inputs through the pin's actual prepare, warm-start and solve functions. */
export function nativeJointPhases() {
    const gold = JSON.parse(readFileSync(path, "utf8"));
    let source = `#include "body.h"
#include "joint.h"
#include "physics_world.h"
#include "solver.h"
#include "solver_set.h"
#include <stdio.h>
#include <string.h>
static float F(unsigned bits){float f;memcpy(&f,&bits,4);return f;}
static void value(const char* key,const float* v,int n){printf("\\\"%s\\\":",key);if(n>1)printf("[");for(int i=0;i<n;++i){unsigned u;memcpy(&u,v+i,4);printf("%s\\\"0x%08x\\\"",i?",":"",u);}if(n>1)printf("]");}
int main(void){\n`;
    const membership: { family: string; index: number; phase: string }[] = [];
    for (const family of families) for (const [index, scenario] of gold[family].entries()) {
        const input = scenario.input;
        const kind = family[0].toUpperCase() + family.slice(1);
        source += `{b3World world={0};b3StepContext ctx={0};b3Body bodies[2]={0};b3BodySim sims[2]={0};b3BodyState states[2]={0};b3SolverSet sets[3]={0};b3JointSim base={0};
world.bodies.data=bodies;world.bodies.count=2;world.solverSets.data=sets;world.solverSets.count=3;
sets[b3_awakeSet].bodySims.data=sims;sets[b3_staticSet].bodySims.data=sims;
ctx.world=&world;ctx.states=states;ctx.sims=sims;
base.type=b3_${family}Joint;base.bodyIdA=0;base.bodyIdB=1;
bodies[0].setIndex=b3_awakeSet;bodies[1].setIndex=${input.bStatic ? "b3_staticSet" : "b3_awakeSet"};bodies[1].localIndex=1;\n`;
        source += assign("ctx.h", input.h) + assign("ctx.inv_h", input.invH) + assign("ctx.enableWarmStarting", input.enableWarmStarting);
        source += assign("base.constraintHertz", input.constraintHertz) + assign("base.constraintDampingRatio", input.constraintDamping);
        for (const [i, side] of ["A", "B"].entries()) {
            source += `states[${i}].flags=${i === 1 && input.bStatic ? "0" : "b3_dynamicFlag"};`;
            for (const [field, key] of [["invMass", "invMass"], ["invInertiaWorld", "invI"], ["transform.q", "q"], ["localCenter", "localCenter"]])
                source += assign(`sims[${i}].${field}`, input[key + side]);
            const center = input["center" + side].map((v: string) => f(v));
            source += `sims[${i}].center=(b3Pos){${center.join(",")}};`;
            source += assign(`base.localFrame${side}.p`, input[`localFrame${side}p`]) + assign(`base.localFrame${side}.q`, input[`localFrame${side}q`]);
            for (const [field, key] of [["linearVelocity", "v"], ["angularVelocity", "w"], ["deltaPosition", "dp"], ["deltaRotation", "dq"]])
                source += assign(`states[${i}].${field}`, input[key + side]);
        }
        const common = new Set(["h", "invH", "enableWarmStarting", "bStatic", "constraintHertz", "constraintDamping"]);
        for (const [key, value] of Object.entries(input)) {
            if (common.has(key) || /^(invMass|invI|q|localCenter|center|localFrame|v[AB]|w[AB]|dp|dq)/.test(key)) continue;
            source += assign(`base.${family}Joint.${key}`, value as string | number | string[]);
        }
        source += "b3PrepareJoint(&base,&ctx);";
        for (const phase of ["warmStart", "solveBias", "solveRelax"]) {
            source += phase === "warmStart" ? `b3WarmStart${kind}Joint(&base,&ctx);` : `b3Solve${kind}Joint(&base,&ctx${family === "motor" ? "" : `,${phase === "solveBias"}`} );`;
            source += 'printf("{");';
            let comma = false;
            for (const [key, value] of Object.entries(scenario[phase])) {
                if (comma) source += 'printf(",");';
                comma = true;
                const velocity = /^([vw])([AB])$/.exec(key);
                const field = velocity ? `states[${velocity[2] === "A" ? 0 : 1}].${velocity[1] === "v" ? "linearVelocity" : "angularVelocity"}` : `base.${family}Joint.${key}`;
                source += `value("${key}",(float*)&${field},${Array.isArray(value) ? value.length : 1});`;
            }
            source += 'printf("}\\n");';
            membership.push({ family, index, phase });
        }
        source += "}\n";
    }
    source += "return 0;}";
    const dir = mkdtempSync(join(tmpdir(), "joint-phases-"));
    try {
        const file = join(dir, "phases.c");
        writeFileSync(file, source);
        const binary = nativeBinary(file, "x86_64");
        const output = run([binary]);
        expect(run([binary])).toBe(output);
        return { gold, membership, outputs: output.trim().split("\n").map(line => JSON.parse(line)) };
    } finally { rmSync(dir, { recursive: true, force: true }); }
}

test("joint phase golds come from the pin's native functions", () => {
    const { gold, membership, outputs } = nativeJointPhases();
    for (const [i, { family, index, phase }] of membership.entries())
        expect(gold[family][index][phase], `${family} ${index} ${phase}`).toEqual(outputs[i]);
});
