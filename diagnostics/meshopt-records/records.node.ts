import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { loadRecords, assertDecoded, referenceOutput } from "./records";

setDefaultTimeout(TIMEOUT.node);

// Failure diagnostics alone load the pinned oracle. The normal record checks do not.
test("a decoded record mismatch names the view and first differing byte", async () => {
    const record = (await loadRecords())[0];
    const output = await referenceOutput(record);
    output[0] ^= 1;
    await expect(assertDecoded(record, output)).rejects.toThrow(`${record.asset} bufferView ${record.index}: SHA-256 mismatch`);
    await expect(assertDecoded(record, output)).rejects.toThrow("first differing offset 0");
});
