import * as d from "typegpu/data";

d.textureStorage2d("rgba16float", "write-only");

// A misspelled storage format otherwise resolves to `texture_storage_2d<undefined, write>` without
// throwing. Keep the format a closed type boundary so `tsc`, which runs in `bun check`, rejects it.
// @ts-expect-error `rgba16floatx` is not a storage texture format
d.textureStorage2d("rgba16floatx", "write-only");
