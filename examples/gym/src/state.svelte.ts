import type { State } from "@dylanebert/shallot";
import type { GymInit } from "./bridge";

export const gymState: { ecs: State | null } = $state({ ecs: null });

let _initCb: ((data: GymInit) => void) | null = null;

export function setInitCallback(cb: (data: GymInit) => void): void {
    _initCb = cb;
}

export function fireInit(data: GymInit): void {
    _initCb?.(data);
}
