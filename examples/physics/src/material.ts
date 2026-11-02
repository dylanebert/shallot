import { Sleeping } from "@dylanebert/shallot";
import {
    applyPbrLighting,
    extendedMaterial,
    instanceData,
    MaterialPlugin,
    Materials,
    pbrInputFromStandardMaterial,
} from "@dylanebert/shallot/standard/rendering";
import * as d from "typegpu/data";

const Sleep = instanceData(Sleeping);
const Parameters = d.struct({ restingColor: d.vec3f });
export const ShapeMaterial = extendedMaterial(Materials, {
    name: "PhysicsShape",
    parameters: Parameters,
    defaults: { restingColor: d.vec3f(0.52, 0.29, 0.065) },
    instanceData: [Sleep],
    fragment: (ctx, parameters) => {
        "use gpu";
        const input = pbrInputFromStandardMaterial(ctx);
        if (Sleep.present(ctx)) {
            let color = d.vec3f(parameters.restingColor);
            if (input.baseColor.r > 0.9) color = d.vec3f(0.65, 0.16, 0.045);
            input.baseColor = d.vec4f(color, input.baseColor.a);
        }
        return applyPbrLighting(input);
    },
});
export const ShapeMaterialPlugin = MaterialPlugin(ShapeMaterial);
