export default `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Sample Viewer capture</title>
<style>html,body{margin:0}canvas{display:block}</style></head><body>
<canvas width="960" height="720"></canvas>
<script type="module">
import { GltfView, ResourceLoader } from '/dist/gltf-viewer.module.js';
const canvas = document.querySelector('canvas');
const context = canvas.getContext('webgl2', {alpha:false, antialias:true});
const info = context.getExtension('WEBGL_debug_renderer_info');
const renderer = info && context.getParameter(info.UNMASKED_RENDERER_WEBGL);
const view = new GltfView(context);
// These scenes use neither Draco nor KTX; do not initialize their optional decoders.
const loader = new ResourceLoader(view);
const state = view.createState();
state.renderingParameters.useDirectionalLightsWithDisabledIBL = true;
state.environment = await loader.loadEnvironment('/assets/environments/footprint_court.hdr');
state.gltf = await loader.loadGltf(new URLSearchParams(location.search).get('model'));
state.sceneIndex = state.gltf.scene ?? 0;
state.cameraIndex = undefined;
state.gltf.scenes[state.sceneIndex].applyTransformHierarchy(state.gltf);
state.userCamera.aspectRatio = canvas.width / canvas.height;
state.userCamera.fitViewToScene(state.gltf, state.sceneIndex);
state.animationIndices = [];
view.renderFrame(state, canvas.width, canvas.height);
window.capture = {
    renderer,
    position: Array.from(state.userCamera.getPosition()),
    fov: state.userCamera.yfov * 180 / Math.PI,
    near: state.userCamera.znear,
    far: state.userCamera.zfar,
};
</script></body></html>
`;
