import { minimalLight, type Config } from "@dylanebert/shallot";
import { TweenPlugin, OrbitPlugin } from "@dylanebert/shallot/extras";

export const config: Config = {
    plugins: [TweenPlugin, OrbitPlugin],
    scene: "/tween/scenes/tween.scene",
    loading: minimalLight(),
};
