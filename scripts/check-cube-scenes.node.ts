import { expect, setDefaultTimeout, test } from "bun:test";
import { readFileSync } from "node:fs";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

const paths = [
    "examples/minimal/src/welcome/index.ts",
    "examples/hud/src/turntable/index.ts",
    "examples/loading-screen/src/welcome/index.ts",
    "examples/embedded-scene/src/scene.ts",
];

function scene(source: string) {
    const literal = (pattern: RegExp) => {
        const match = source.match(pattern);
        if (!match) throw new Error(`Missing scene literal: ${pattern}`);
        return match[1]!.replace(/\s/g, "");
    };
    const vector = (entity: string, field: string) =>
        JSON.parse(
            literal(
                new RegExp(`world\\.add\\(${entity}, Transform, \\{[^}]*${field}: (\\[[^\\]]+\\])`),
            ),
        ) as number[];
    return {
        ambientColor: literal(/ambient.color = (\w+)/),
        ambientBrightness: literal(/ambient.brightness = ([\d.]+)/),
        sunDirection: literal(/rotation: lookAtRotation\(([^)]+)\)/),
        sun: literal(/world.add\(sun, DirectionalLight, (\{[^}]+\})/),
        shadows: literal(/shadowMapsEnabled.set\(sun, (\d)\)/),
        camera: literal(/world.add\(camera, Camera, (\{[^}]+\})/),
        orbit: literal(/world.add\(camera, Orbit, (\{[^}]+\})/),
        materials: [...source.matchAll(/StandardMaterial\((\{[^}]+\})/g)].map((match) =>
            match[1]!.replace(/\s/g, ""),
        ),
        cube: vector("cube", "translation"),
        ground: vector("ground", "translation"),
        scale: vector("ground", "scale"),
    };
}

test("the four copied cube scenes have equal geometry, orbit, lighting and materials", () => {
    const scenes = paths.map((path) => scene(readFileSync(path, "utf8")));
    for (const [index, value] of scenes.entries()) expect(value, paths[index]).toEqual(scenes[0]!);
});

test.each(paths)("%s rests its unit cube on the platform", (path) => {
    const value = scene(readFileSync(path, "utf8"));
    expect(
        Math.abs(value.cube[1]! - 0.5 - (value.ground[1]! + value.scale[1]! / 2)),
    ).toBeLessThanOrEqual(1e-4);
});
