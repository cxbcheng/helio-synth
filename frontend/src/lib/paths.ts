export const DATASET_NAME = "hmi.v_45s.res512_cadence45s";
export const N_POINTS = 4000;
const DATA_ROOT = import.meta.env.VITE_DATA_BASE_URL ?? "/data";
export const DATASET_BASE = `${DATA_ROOT}/${DATASET_NAME}`;


export function metadataUrl(): string {
    return new URL(`${DATASET_BASE}/metadata.json`, window.location.origin).toString();
}

export function zarrStoreUrl(): string {
    return new URL(`${DATASET_BASE}/timeseries/n${N_POINTS}.zarr`, window.location.origin).toString();
}

/**
 * Based on rendered frame filenames "<YYYYMMDD>_<HHMMSS>.webp"
 */
export function imageUrl(timestampTag: string, colormap: string, detrendOrder: number | null): string {
    const orderName = detrendOrder !== null ? detrendOrder.toString() : 'raw';
    return `${DATASET_BASE}/images/${colormap}/${orderName}/${timestampTag}.webp`;
}
