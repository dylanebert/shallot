export {
    AudioPlugin,
    type InstrumentDef,
    Instruments,
    instrument,
    Listener,
    type ModulationDef,
    type NodeDef,
    type NodeType,
    play,
    type SfxPolicy,
    Sound,
    sample,
    sfx,
} from "../transitional/audio";
export { BvhPlugin } from "../transitional/bvh";
export { Character, CharacterPlugin, CharacterSweepSystem } from "../transitional/character";
export { Glaze, GlazePlugin, Tonemap } from "../transitional/glaze";
export { Mirror, MirrorPlugin, MirrorSystem, mirror } from "../transitional/mirror";
export { Color, Part, PartPlugin } from "../transitional/part";
export {
    Body,
    BodyType,
    body,
    type ContactEvents,
    createParallelJoint,
    createRevoluteJoint,
    createSoftJoint,
    createSphericalJoint,
    createWheelJoint,
    getContactEvents,
    getJointEvents,
    hash,
    Joint,
    JointType,
    type ParallelJointConfig,
    Physics,
    PhysicsPlugin,
    physicsCounters,
    physicsStepConfig,
    physicsWorld,
    type RevoluteJointConfig,
    readBody,
    restore,
    ShapeKind,
    SoftJoint,
    type SoftJointConfig,
    type SphericalJointConfig,
    Spring,
    setKinematic,
    setVelocity,
    snapshot,
    type WheelJointConfig,
    World,
} from "../transitional/physics";
export { Slab, SlabPlugin, SlabSystem, slab } from "../transitional/slab";
export { composeTransform, Transform, TransformsPlugin } from "../transitional/transforms";
export {
    type LoadingOptions,
    minimalDark,
    minimalLight,
    type SplashOptions,
    type SplashProfile,
    shallotDark,
    shallotLight,
} from "./loading";
export {
    Backdrop,
    Depth,
    MAX_CASCADES,
    MAX_POINT_CASTERS,
    Material,
    PointShadows,
    Sear,
    SearPlugin,
    Shadow,
    SunShadows,
    Tag,
} from "./rendering";

import { BrowserInputPlugin, InputPlugin } from "../core/input";
import { RenderPlugin } from "../core/rendering";
import type { Plugin } from "../engine";
import { setDefaultLoading, setDefaultPlugins } from "../engine/app";
import { GlazePlugin } from "../transitional/glaze";
import { PartPlugin } from "../transitional/part";
import { SlabPlugin } from "../transitional/slab";
import { TransformsPlugin } from "../transitional/transforms";
import { shallotDark } from "./loading";
import { SearPlugin } from "./rendering";

export const DEFAULT_PLUGINS: readonly Plugin[] = [
    SlabPlugin,
    TransformsPlugin,
    InputPlugin,
    BrowserInputPlugin,
    RenderPlugin,
    PartPlugin,
    SearPlugin,
    GlazePlugin,
];

setDefaultPlugins(DEFAULT_PLUGINS);
setDefaultLoading(shallotDark);
