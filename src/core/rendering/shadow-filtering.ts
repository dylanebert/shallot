import { component, u32 } from "../../engine";

/** The available standard shadow filters. Hardware2x2 uses one comparison sample; Gaussian is the default. */
export const ShadowFilteringMethodMode = {
    Hardware2x2: 0,
    Gaussian: 1,
} as const;

/** Shadow edge filter selected per camera. A camera without this component uses Gaussian. */
export const ShadowFilteringMethod = component(
    "ShadowFilteringMethod",
    { method: u32 },
    { defaults: () => ({ method: ShadowFilteringMethodMode.Gaussian }) },
);
