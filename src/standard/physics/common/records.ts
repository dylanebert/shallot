export type Vec3 = { x: number; y: number; z: number };

/** Quaternion vector part `v` and scalar part `s`. */
export type Quat = { v: Vec3; s: number };

/** Rotation `q` followed by translation `p`. */
export type Transform = { p: Vec3; q: Quat };

/** Column-major 3×3 matrix. */
export type Mat3 = { cx: Vec3; cy: Vec3; cz: Vec3 };
export type AABB = { lowerBound: Vec3; upperBound: Vec3 };

/** Plane separation is dot(normal, point) − offset. */
export type Plane = { normal: Vec3; offset: number };
export type Pos = Vec3;
export type WorldTransform = Transform;
