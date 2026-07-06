import { TweenPlugin } from "@dylanebert/shallot";
import { start } from "./boot";

// Parallel + sequential + loop, authored entirely in the scene. Five cubes rise together then fall
// together on one looping timeline (ten `<a tween>` placed at `at: 0` and `at: 1.4` on one `<a sequence>`),
// each cube carrying its own easing curve. A retained reference line marks the top the springy curves
// overshoot — lines + tween composited, the point of the visualization set. No script: the scene is the demo.
await start([TweenPlugin], "../scenes/tween.scene");
