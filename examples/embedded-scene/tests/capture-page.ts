import { captureFrame } from "@dylanebert/shallot/rendering";

const page = window as Window & {
    shallotCaptureFrame?: typeof captureFrame;
};
page.shallotCaptureFrame = captureFrame;
