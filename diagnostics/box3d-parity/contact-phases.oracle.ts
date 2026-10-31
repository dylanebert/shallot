import { expect, test } from 'bun:test';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { nativeBinary, run } from './native';

const root = resolve('src/standard/physics/collision');
const f = (bits: string) => `F(${bits})`;
const fields = (name: string, values: string[]) => values.map((v, i) => `${name}[${i}]=${f(v)};`).join('\n');
const vec = (name: string, values: string[]) => `${name}=(b3Vec3){${values.map(f).join(',')}};`;
const quat = (name: string, values: string[]) => `${name}=(b3Quat){{${values.slice(0, 3).map(f).join(',')}},${f(values[3]!)}};`;
const base = `b3World world={0};b3StepContext ctx={0};b3Contact contacts[4]={0};b3Manifold manifolds[4]={0};b3BodySim sims[8]={0};b3BodyState states[8]={0};b3TaskContext task={0};uint64_t hits=0;setup(&world,&ctx,contacts,4,sims,states,&task,&hits);`;
const setFloat = (name: string, value: string) => `${name}=${f(value)};`;
const softness = (name: string, value: string[]) => `${name}=(b3Softness){${value.map(f).join(',')}};`;

// Inputs stay in their historical JSON representation; only the native outputs are replaced.
export function nativeContactPhases() {
    const scalarGold = JSON.parse(readFileSync(join(root, 'contact.gold.json'), 'utf8'));
    const wideGold = JSON.parse(readFileSync(join(root, 'contact_wide.gold.json'), 'utf8'));
    let source = readFileSync(resolve('diagnostics/box3d-parity/contact-phases.c'), 'utf8') + '\nint main(void){\n';
    for (const c of scalarGold.prepare) {
        source += `{${base}world.contacts.count=1;world.enableWarmStarting=${c.warmStartScale !== '0x00000000'};\n`;
        source += softness('ctx.contactSoftness', c.contactSoftness) + softness('ctx.staticSoftness', c.staticSoftness);
        source += setFloat('ctx.inv_h', c.invH) + setFloat('world.contactSpeed', c.contactSpeed);
        source += setFloat('world.restitutionThreshold', c.restThreshold) + setFloat('world.hitEventThreshold', c.hitThreshold);
        for (let i = 0; i < 2; i++) {
            const s = c.state.slice(i * 16, i * 16 + 16), sim = c.sim.slice(i * 32, i * 32 + 32);
            source += vec(`states[${i}].linearVelocity`, s.slice(0, 3)) + vec(`states[${i}].angularVelocity`, s.slice(3, 6));
            source += vec(`states[${i}].deltaPosition`, s.slice(6, 9)) + quat(`states[${i}].deltaRotation`, s.slice(9, 13));
            source += `states[${i}].flags=b3_dynamicFlag;` + setFloat(`sims[${i}].invMass`, sim[0]);
            source += fields(`((float*)&sims[${i}].invInertiaWorld)`, sim.slice(19, 28));
        }
        source += `contacts[0].encodedBodySimA=${c.inContactMeta[0] === 4294967295 ? -1 : c.inContactMeta[0]};contacts[0].encodedBodySimB=${c.inContactMeta[1] === 4294967295 ? -1 : c.inContactMeta[1]};contacts[0].flags=${c.inContactMeta[4]};contacts[0].manifoldCount=1;contacts[0].manifolds=manifolds;`;
        source += setFloat('contacts[0].friction', c.inContact[0]) + setFloat('contacts[0].restitution', c.inContact[1]) + setFloat('contacts[0].rollingResistance', c.inContact[2]) + vec('contacts[0].tangentVelocity', c.inContact.slice(3, 6));
        source += vec('manifolds[0].normal', c.inManifold.slice(0, 3)) + vec('manifolds[0].frictionImpulse', c.inManifold.slice(3, 6)) + setFloat('manifolds[0].twistImpulse', c.inManifold[6]) + vec('manifolds[0].rollingImpulse', c.inManifold.slice(7, 10));
        source += `manifolds[0].pointCount=${c.pointCount};`;
        for (let p = 0; p < c.pointCount; p++) {
            const v = c.inPoint.slice(p * 10, p * 10 + 10), n = `manifolds[0].points[${p}]`;
            source += vec(n + '.anchorA', v.slice(0, 3)) + vec(n + '.anchorB', v.slice(3, 6)) + setFloat(n + '.separation', v[6]) + setFloat(n + '.normalImpulse', v[7]);
        }
        source += 'capture_scalar(&ctx);}\n';
    }
    source += `{${base}world.enableWarmStarting=true;`;
    source += softness('ctx.contactSoftness', wideGold.softness.cs) + softness('ctx.staticSoftness', wideGold.softness.ss) + setFloat('ctx.inv_h', wideGold.invH) + setFloat('world.contactSpeed', wideGold.contactSpeed) + setFloat('world.restitutionThreshold', wideGold.restitutionThreshold) + setFloat('world.hitEventThreshold', wideGold.hitThreshold);
    let body = 0;
    for (let i = 0; i < 4; i++) {
        const c = wideGold.lanes[i].input;
        for (const side of ['A', 'B']) {
            const index = side === 'B' && c.bNull ? -1 : body++;
            source += `contacts[${i}].encodedBodySim${side}=${index};`;
            if (index < 0) continue;
            source += setFloat(`sims[${index}].invMass`, c['invMass' + side]) + fields(`((float*)&sims[${index}].invInertiaWorld)`, c['invI' + side]);
            source += vec(`states[${index}].linearVelocity`, c['v' + side]) + vec(`states[${index}].angularVelocity`, c['w' + side]) + vec(`states[${index}].deltaPosition`, c['dp' + side]) + quat(`states[${index}].deltaRotation`, c['dq' + side]) + `states[${index}].flags=b3_dynamicFlag;`;
        }
        source += `contacts[${i}].contactId=${i};contacts[${i}].flags=${c.flags};contacts[${i}].manifoldCount=1;contacts[${i}].manifolds=manifolds+${i};`;
        for (const field of ['friction', 'restitution', 'rollingResistance']) source += setFloat(`contacts[${i}].${field}`, c[field]);
        source += vec(`contacts[${i}].tangentVelocity`, c.tangentVelocity);
        source += vec(`manifolds[${i}].normal`, c.normal) + vec(`manifolds[${i}].frictionImpulse`, c.mFriction) + setFloat(`manifolds[${i}].twistImpulse`, c.mTwist) + vec(`manifolds[${i}].rollingImpulse`, c.mRolling) + `manifolds[${i}].pointCount=${c.pointCount};`;
        for (let p = 0; p < c.pointCount; p++) {
            const n = `manifolds[${i}].points[${p}]`;
            source += vec(n + '.anchorA', c.rA[p]) + vec(n + '.anchorB', c.rB[p]) + setFloat(n + '.separation', c.separation[p]) + setFloat(n + '.normalImpulse', c.mpNormalImpulse[p]);
        }
    }
    source += 'capture_wide(&ctx);}\nreturn 0;}';
    const dir = mkdtempSync(join(tmpdir(), 'contact-phases-'));
    try {
        const file = join(dir, 'phases.c');writeFileSync(file, source);
        const outputs = run([nativeBinary(file, 'x86_64')]).trim().split('\n').map((line) => JSON.parse(line));
        return { scalarGold, wideGold, outputs };
    } finally { rmSync(dir, { recursive: true, force: true }); }
}

test('scalar and SSE2-wide contact phase golds come from the pinned native functions', () => {
    const { scalarGold, wideGold, outputs } = nativeContactPhases();
    for (let i = 0; i < scalarGold.prepare.length; i++) for (const [key, value] of Object.entries(outputs[i])) expect(scalarGold.prepare[i][key], `scalar ${i} ${key}`).toEqual(value);
    for (let i = 0; i < 4; i++) for (const [key, value] of Object.entries(outputs[i + scalarGold.prepare.length])) expect(wideGold.lanes[i][key], `wide ${i} ${key}`).toEqual(value);
});
