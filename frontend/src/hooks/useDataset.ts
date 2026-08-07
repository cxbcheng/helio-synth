import { useEffect, useState } from "react";
import { loadDataset, type Dataset } from "../lib/zarr";
import {metadataUrl} from "../lib/paths";

export interface Metadata {
    width: number;
    height: number;
    cadence: number;
    images: { colormaps: string[]; detrend_orders: number[] };
    timeseries: { samples: number[] };
}

interface State {
    metadata: Metadata | null;
    dataset: Dataset | null;
    loading: boolean;
    error: string | null;
}

export function useDataset(): State {
    const [state, setState] = useState<State>({
        metadata: null,
        dataset: null,
        loading: true,
        error: null,
    });

    useEffect(() => {
        let canceled = false;

        (async () => {
            try {
                console.log(metadataUrl());

                const metaRes = await fetch(metadataUrl());
                if (!metaRes.ok) throw new Error(`metadata.json fetch failed: ${metaRes.status}`);
                const metadata: Metadata = await metaRes.json();
                const dataset = await loadDataset();
                if (!canceled) setState({ metadata, dataset, loading: false, error: null });
            } catch (err) {
                if (!canceled) {
                    setState({
                        metadata: null,
                        dataset: null,
                        loading: false,
                        error: err instanceof Error ? err.message : String(err),
                    });
                }
                throw err;
            }
        })();

        return () => {
            canceled = true;
        };
    }, []);

    return state;
}
