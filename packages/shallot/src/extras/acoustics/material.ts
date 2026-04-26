import { traits, buf, CHUNK_MASK, CHUNK_SHIFT } from "../../engine";
import { createFieldProxy } from "../../engine/ecs/core";

export const AcousticMaterialData = buf(Float32Array, 8, 0);

export type MaterialValues = {
    absorptionLow: number;
    absorptionMid: number;
    absorptionHigh: number;
    scattering: number;
    transmissionLow: number;
    transmissionMid: number;
    transmissionHigh: number;
};

function mat(
    al: number,
    am: number,
    ah: number,
    sc: number,
    tl: number,
    tm: number,
    th: number,
): MaterialValues {
    return {
        absorptionLow: al,
        absorptionMid: am,
        absorptionHigh: ah,
        scattering: sc,
        transmissionLow: tl,
        transmissionMid: tm,
        transmissionHigh: th,
    };
}

const presets = [
    mat(0.1, 0.2, 0.3, 0.05, 0.1, 0.05, 0.03),
    mat(0.03, 0.04, 0.07, 0.05, 0.015, 0.015, 0.015),
    mat(0.05, 0.07, 0.08, 0.05, 0.015, 0.002, 0.001),
    mat(0.01, 0.02, 0.02, 0.05, 0.06, 0.044, 0.011),
    mat(0.6, 0.7, 0.8, 0.05, 0.031, 0.012, 0.008),
    mat(0.24, 0.69, 0.73, 0.05, 0.02, 0.005, 0.003),
    mat(0.06, 0.03, 0.02, 0.05, 0.06, 0.044, 0.011),
    mat(0.12, 0.06, 0.04, 0.05, 0.056, 0.056, 0.004),
    mat(0.11, 0.07, 0.06, 0.05, 0.07, 0.014, 0.005),
    mat(0.2, 0.07, 0.06, 0.05, 0.2, 0.025, 0.01),
    mat(0.13, 0.2, 0.24, 0.05, 0.015, 0.002, 0.001),
];

export const MaterialPreset = {
    Generic: 0,
    Brick: 1,
    Concrete: 2,
    Ceramic: 3,
    Gravel: 4,
    Carpet: 5,
    Glass: 6,
    Plaster: 7,
    Wood: 8,
    Metal: 9,
    Rock: 10,
} as const;

function applyPreset(eid: number, id: number) {
    const p = presets[id];
    if (!p) return;
    const chunk = AcousticMaterialData.chunks[eid >>> CHUNK_SHIFT];
    const o = (eid & CHUNK_MASK) * 8;
    chunk[o] = p.absorptionLow;
    chunk[o + 1] = p.absorptionMid;
    chunk[o + 2] = p.absorptionHigh;
    chunk[o + 3] = p.scattering;
    chunk[o + 4] = p.transmissionLow;
    chunk[o + 5] = p.transmissionMid;
    chunk[o + 6] = p.transmissionHigh;
    chunk[o + 7] = id;
}

const presetProxy = new Proxy([] as number[], {
    set(target, prop, value) {
        if (typeof prop === "string") {
            const eid = Number(prop);
            if (eid >= 0) applyPreset(eid, value);
        }
        return Reflect.set(target, prop, value);
    },
    get(target, prop) {
        if (typeof prop === "string") {
            const eid = Number(prop);
            if (eid >= 0) {
                const chunk = AcousticMaterialData.chunks[eid >>> CHUNK_SHIFT];
                return chunk[(eid & CHUNK_MASK) * 8 + 7];
            }
        }
        return Reflect.get(target, prop);
    },
});

export const AcousticMaterial = {
    preset: presetProxy as unknown as number[],
    absorptionLow: createFieldProxy(AcousticMaterialData, 8, 0),
    absorptionMid: createFieldProxy(AcousticMaterialData, 8, 1),
    absorptionHigh: createFieldProxy(AcousticMaterialData, 8, 2),
    scattering: createFieldProxy(AcousticMaterialData, 8, 3),
    transmissionLow: createFieldProxy(AcousticMaterialData, 8, 4),
    transmissionMid: createFieldProxy(AcousticMaterialData, 8, 5),
    transmissionHigh: createFieldProxy(AcousticMaterialData, 8, 6),
};

traits(AcousticMaterial, {
    defaults: () => ({ preset: MaterialPreset.Generic }),
    enums: { preset: MaterialPreset as unknown as Record<string, number> },
});
