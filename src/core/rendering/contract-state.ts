import { Registry } from "../../engine";
import type { Background, Surface } from "./contract";

/** every schema-backed surface, keyed by name with a stable renderer-owned id. */
export const surfacesKey = { create: () => new Registry<Surface>() };

/** every registered background, keyed by name. */
export const backgroundsKey = { create: () => new Registry<Background>() };
