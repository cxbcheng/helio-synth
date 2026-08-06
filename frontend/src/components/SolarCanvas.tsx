import { useEffect, useRef, useState } from "react";

interface Props {
    width: number;
    height: number;
    imageUrl: string | null;
}

// Decoded-frame cache so scrubbing back and forth doesn't re-decode the
// same WebP repeatedly. Individually cached pre-rendered frames + an
// ImageBitmap cache is the pattern that tested fastest for hover/scrub
// responsiveness (vs. sprite sheets or video-frame extraction).
const bitmapCache = new Map<string, ImageBitmap>();

export function SolarCanvas({ width, height, imageUrl }: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!imageUrl) return;
        let cancelled = false;

        (async () => {
            try {
                let bitmap = bitmapCache.get(imageUrl);
                if (!bitmap) {
                    const res = await fetch(imageUrl);
                    if (!res.ok) throw new Error(`Frame fetch failed (${res.status}): ${imageUrl}`);
                    bitmap = await createImageBitmap(await res.blob());
                    bitmapCache.set(imageUrl, bitmap);
                }
                if (cancelled) return;

                const ctx = canvasRef.current?.getContext("2d");
                if (!ctx || !canvasRef.current) return;
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                ctx.drawImage(bitmap, 0, 0, canvasRef.current.width, canvasRef.current.height);
                setError(null);
            } catch (err) {
                if (!cancelled) setError(err instanceof Error ? err.message : String(err));
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [imageUrl]);

    return (
        <div>
            <canvas ref={canvasRef} width={width} height={height} />
            {error && <p style={{ color: "red" }}>{error}</p>}
        </div>
    );
}
