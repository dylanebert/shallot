import { createApp } from "@dylanebert/shallot/app";
import { AudioOutput, AudioPlugin, play, registerClip } from "@dylanebert/shallot/audio";
import { StandardAudioPlugin } from "@dylanebert/shallot/standard/audio";

const status = document.querySelector("#status")!;
const app = await createApp({ defaults: false, plugins: [AudioPlugin, StandardAudioPlugin] });
const world = app.world;
const context = world.resource(AudioOutput).context;
if (!context) throw new Error("StandardAudioPlugin did not publish its AudioContext");

const pcm = new Float32Array(context.sampleRate / 2);
for (let i = 0; i < pcm.length; i++)
    pcm[i] = Math.sin((2 * Math.PI * 440 * i) / context.sampleRate) * 0.25;
const clip = registerClip(world, pcm, context.sampleRate);
play(world, clip, { loop: true });

const analyser = context.createAnalyser();
analyser.fftSize = 2048;
world.resource(AudioOutput).bus!.connect(analyser);
analyser.connect(context.destination);
const samples = new Float32Array(analyser.fftSize);
(window as Window & { audioPeak?: () => number }).audioPeak = () => {
    analyser.getFloatTimeDomainData(samples);
    let peak = 0;
    for (const value of samples) peak = Math.max(peak, Math.abs(value));
    return peak;
};
document.querySelector("button")!.addEventListener("click", () => {
    status.textContent = "playing";
});
status.textContent = "ready";
const frame = () => {
    world.step(1 / 60);
    requestAnimationFrame(frame);
};
requestAnimationFrame(frame);
