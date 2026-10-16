import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    NotShadowCaster,
    ShadowFilteringMethod,
    ShadowFilteringMethodMode,
    SpotLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { spotParams } from "./lighting";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";
import { PointShadows, pointFaceVP } from "./shadows";

setDefaultTimeout(TIMEOUT.node);
const methods = [ShadowFilteringMethodMode.Hardware2x2, ShadowFilteringMethodMode.Gaussian];
const subject = gpuApps(
    import.meta.path,
    Array.from({ length: methods.length }, () => ({
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin],
    })),
);

for (const method of methods) {
    test(`wide cone occluder darkens the 85 degree receiver: ${method}`, async () => {
        const { world } = subject()[method]!;
        console.log("adapter", world.gpu.adapter);
        world.resource(PointShadows).atlas = 4096;
        const camera = world.create();
        const z = -6 / Math.tan((85 * Math.PI) / 180);
        world.add(camera, Transform, {
            translation: [2, 0, 3, 0],
            rotation: lookAtRotation(2, 0, 3, 6, 0, z),
        });
        world.add(camera, Camera);
        world.storage(Camera).antialias.set(camera, 0);
        world.add(camera, AmbientLight, { brightness: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, ShadowFilteringMethod, { method });
        attachTexture(world, camera, { width: 128, height: 128 });
        const material = world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [0.8, 0.8, 0.8, 1] }));
        const receiver = world.create();
        world.add(receiver, Transform, { translation: [6.1, 0, z, 0], scale: [0.2, 3, 3, 0] });
        world.add(receiver, MeshInstance);
        world.add(receiver, MeshMaterial, material);
        const occluder = world.create();
        world.add(occluder, Transform, { translation: [3, 0, z / 2, 0], scale: [0.3, 2, 0.2, 0] });
        world.add(occluder, MeshInstance);
        world.add(occluder, MeshMaterial, material);
        const light = world.create();
        world.add(light, Transform);
        world.add(light, SpotLight, {
            intensity: 100000,
            range: 20,
            innerAngle: (88 * Math.PI) / 180,
            outerAngle: (89 * Math.PI) / 180,
            shadowMapsEnabled: 1,
        });
        world.add(occluder, NotShadowCaster);
        world.step(0);
        world.step(0);
        const control = (await captureTexture(world, camera)).rgba;
        world.remove(occluder, NotShadowCaster);
        world.step(0);
        world.step(0);
        const shadow = (await captureTexture(world, camera)).rgba;
        const mean = (bytes: Uint8ClampedArray) => {
            let sum = 0;
            for (let y = 60; y < 68; y++)
                for (let x = 60; x < 68; x++) sum += bytes[(y * 128 + x) * 4];
            return sum / 64;
        };
        console.log({ method, control: mean(control), shadow: mean(shadow) });
        expect(mean(control)).toBeGreaterThan(40);
        expect(mean(shadow)).toBeLessThan(mean(control) * 0.5);
        const wideProjection = pointFaceVP(world).slice(0, 16);
        const packedParams = async () =>
            new Float32Array(
                (
                    await probeBuffer(world, world.gpu.buffers.get("lightClusters")!, {
                        offset: 56,
                        size: 8,
                    })
                ).bytes,
            );
        const wideParams = await packedParams();
        const angles = world.storage(SpotLight);
        angles.innerAngle.set(light, (20 * Math.PI) / 180);
        angles.outerAngle.set(light, (30 * Math.PI) / 180);
        world.step(0);
        world.step(0);
        expect(pointFaceVP(world).slice(0, 16)).not.toEqual(wideProjection);
        expect(await packedParams()).not.toEqual(wideParams);
        angles.innerAngle.set(light, (88 * Math.PI) / 180);
        angles.outerAngle.set(light, (89 * Math.PI) / 180);
        world.step(0);
        world.step(0);
        expect(pointFaceVP(world).slice(0, 16)).toEqual(wideProjection);
        expect(await packedParams()).toEqual(wideParams);
        expect((await captureTexture(world, camera)).rgba).toEqual(shadow);
        expectWarning(/SpotLight outerAngle >= π\/2/);
        angles.innerAngle.set(light, 2);
        angles.outerAngle.set(light, 2);
        world.step(0);
        world.step(0);
        const invalidProjection = pointFaceVP(world).slice(0, 16);
        const invalidParams = await packedParams();
        const effective = Math.fround(Math.PI / 2 - 1e-4);
        const cpu = spotParams(effective, effective);
        expect(invalidParams[0]).toBe(Math.fround(cpu.scale));
        expect(invalidParams[1]).toBe(Math.fround(cpu.offset));
        expect(angles.outerAngle.get(light)).toBe(2);
        expect(angles.innerAngle.get(light)).toBe(2);
        angles.outerAngle.set(light, effective);
        angles.innerAngle.set(light, effective);
        world.step(0);
        world.step(0);
        expect(pointFaceVP(world).slice(0, 16)).toEqual(invalidProjection);
        expect(await packedParams()).toEqual(invalidParams);
        for (const eid of [camera, receiver, occluder, light]) world.destroy(eid);
    });
}
