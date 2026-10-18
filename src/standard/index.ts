export {
    Body,
    BodyMotionLock,
    BodyType,
    DistanceJoint,
    FilterJoint,
    MotorJoint,
    ParallelJoint,
    PrismaticJoint,
    RevoluteJoint,
    ShapeKind,
    SphericalJoint,
    WeldJoint,
    WheelJoint,
} from "../core/physics";
export { composeGlobalTransform, GlobalTransform, Transform } from "../core/transform";
export { loadClip, StandardAudioPlugin } from "./audio";
export {
    type SplashOptions,
    type SplashProfile,
    shallotDark,
} from "./loading";
export {
    applyAngularImpulse,
    applyForce,
    applyForceToCenter,
    applyLinearImpulse,
    applyLinearImpulseToCenter,
    applyTorque,
    Character,
    CharacterPlugin,
    GroundState,
    hashPhysics,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    type StepProfile,
    setAngularVelocity,
    setAwake,
    setLinearVelocity,
    setTargetTransform,
    setTransform,
    shapeMesh,
} from "./physics";
export {
    AlphaMode,
    AlphaModeCode,
    applyPbrLighting,
    CameraBackground,
    DirectionalLightShadowMap,
    extendedMaterial,
    MAX_CASCADES,
    MAX_DIRECTIONAL_LIGHTS,
    MAX_POINT_CASTERS,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    PbrInput,
    PointShadows,
    type ProducerTable,
    pbrInputFromStandardMaterial,
    registerProducerTable,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
    VertexMaterialType,
} from "./rendering";

import { BrowserInputPlugin, InputPlugin } from "../core/input";
import { CorePipelinePlugin, RenderingPlugin } from "../core/rendering";
import type { Plugin } from "../engine";
import { setDefaultLoading, setDefaultPlugins } from "../engine/app";
import { shallotDark } from "./loading";
import { MeshRenderPlugin, StandardRenderingPlugin } from "./rendering";

export const DEFAULT_PLUGINS: readonly Plugin[] = [
    InputPlugin,
    BrowserInputPlugin,
    RenderingPlugin,
    MeshRenderPlugin,
    StandardRenderingPlugin,
    CorePipelinePlugin,
];

setDefaultPlugins(DEFAULT_PLUGINS);
setDefaultLoading(shallotDark);
