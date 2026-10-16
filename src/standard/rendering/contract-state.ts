import type { Background } from "./contract";

/** Registered fullscreen backgrounds owned by the active World. */
export const backgroundsKey = { create: () => new Map<number, Background>() };
