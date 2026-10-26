import type { World } from "@dylanebert/shallot";

export const SceneRestart = {
    create: () => ({
        restart(_construct: (world: World) => number[]): void {
            throw new Error("Scene restart requires sceneSwitcher");
        },
    }),
};
