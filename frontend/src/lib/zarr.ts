import * as zarr from "zarrita";
import { zarrStoreUrl } from "./paths";

export interface Dataset {
    frameTimes: { jd1: Float64Array; jd2: Float64Array }; // length = n_frames
    samplePoints: Float64Array; // flattened (n_points, 2) -- [x0,y0,x1,y1,...]
    velocities: Float32Array; // flattened (n_frames, n_points), row-major
    nFrames: number;
    nPoints: number;
}

/**
 * Loads the entire zarr store into memory.
 * @warning Not feasible for large (multi-GB) datasets. Use time-chunking instead.
 */
export async function loadDataset(): Promise<Dataset> {
    console.warn(
        "Loading full dataset is not feasible for large datasets and is only" +
        "provided for testing. It is recommended to use time-chunking instead."
    );
    const store = new zarr.FetchStore(zarrStoreUrl());
    const root = zarr.root(store);

    const [velArr, jd1Arr, jd2Arr, ptsArr] = await Promise.all([
        zarr.open(root.resolve("velocities"), { kind: "array" }),
        zarr.open(root.resolve("t_jd1"), { kind: "array" }),
        zarr.open(root.resolve("t_jd2"), { kind: "array" }),
        zarr.open(root.resolve("sample_points"), { kind: "array" }),
    ]);

    const [vel, jd1, jd2, pts] = await Promise.all([
        zarr.get(velArr),
        zarr.get(jd1Arr),
        zarr.get(jd2Arr),
        zarr.get(ptsArr),
    ]);

    const [nFrames, nPoints] = vel.shape as [number, number];

    return {
        frameTimes: { jd1: jd1.data as Float64Array, jd2: jd2.data as Float64Array },
        samplePoints: pts.data as Float64Array,
        velocities: vel.data as Float32Array,
        nFrames,
        nPoints,
    };
}
