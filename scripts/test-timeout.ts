import { setDefaultTimeout } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.cheap);
