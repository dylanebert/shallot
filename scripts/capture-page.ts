import { captureFrame } from "@dylanebert/shallot/rendering";

Object.assign(globalThis, { shallotCaptureFrame: captureFrame });
