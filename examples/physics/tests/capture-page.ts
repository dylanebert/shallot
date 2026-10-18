import { captureFrame } from "@dylanebert/shallot/rendering";

(window as Window & { capturePhysicsFrame?: () => Promise<void> }).capturePhysicsFrame =
    async () => {
        const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
        const capture = await captureFrame(canvas);
        const image = document.createElement("canvas");
        image.width = capture.width;
        image.height = capture.height;
        image
            .getContext("2d")!
            .putImageData(
                new ImageData(new Uint8ClampedArray(capture.rgba), capture.width, capture.height),
                0,
                0,
            );
        image.id = "captured-frame";
        image.style.cssText = "position:absolute;inset:0;pointer-events:none";
        canvas.after(image);
    };
