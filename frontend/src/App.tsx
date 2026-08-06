import { useEffect, useMemo, useState } from "react";
import "./App.css";
import { useDataset } from "./hooks/useDataset";
import { SolarCanvas } from "./components/SolarCanvas";
import { imageUrl } from "./lib/paths";
import { timestampTag } from "./lib/time";

function App() {
	const { metadata, dataset, loading, error } = useDataset();

	const [colormap, setColormap] = useState("RdBu_r");
	const [detrendOrder, setDetrendOrder] = useState(2);
	const [frameIndex, setFrameIndex] = useState(0);
	const [playing, setPlaying] = useState(false);

	useEffect(() => {
		if (!metadata) return;
		if (!metadata.images.colormaps.includes(colormap)) {
			setColormap(metadata.images.colormaps[0]);
		}
		if (!metadata.images.detrend_orders.includes(detrendOrder)) {
			setDetrendOrder(metadata.images.detrend_orders[0]);
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [metadata]);

	const frameTags = useMemo(() => {
		if (!dataset) return [];
		const { jd1, jd2 } = dataset.frameTimes;
		return Array.from(jd1, (_, i) => timestampTag(jd1[i], jd2[i]));
	}, [dataset]);

	// Fixed-interval playback for the MVP. Real cadence/alpha-synced
	// timing (get_animation_timestamps + snap_to_available) is a follow-up.
	useEffect(() => {
		if (!playing || frameTags.length === 0) return;
		const id = setInterval(() => setFrameIndex((i) => (i + 1) % frameTags.length), 200);
		return () => clearInterval(id);
	}, [playing, frameTags.length]);

	if (loading) return <p>Loading dataset...</p>;
	if (error) return <p style={{ color: "red" }}>Error: {error}</p>;
	if (!metadata || !dataset || frameTags.length === 0) return <p>No data available.</p>;

	const currentUrl = imageUrl(frameTags[frameIndex], colormap, detrendOrder);

	return (
		<div className="app">
			<h1>HelioSynth</h1>

			<SolarCanvas width={metadata.width} height={metadata.height} imageUrl={currentUrl} />

			<div className="controls">
				<button onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>

				<input
					type="range"
					min={0}
					max={frameTags.length - 1}
					value={frameIndex}
					onChange={(e) => {
						setPlaying(false);
						setFrameIndex(Number(e.target.value));
					}}
				/>
				<span>{frameIndex + 1} / {frameTags.length}</span>

				<select value={colormap} onChange={(e) => setColormap(e.target.value)}>
					{metadata.images.colormaps.map((c) => (
						<option key={c} value={c}>{c}</option>
					))}
				</select>

				<select value={detrendOrder} onChange={(e) => setDetrendOrder(Number(e.target.value))}>
					{metadata.images.detrend_orders.map((o) => (
						<option key={o} value={o}>detrend {o}</option>
					))}
				</select>
			</div>
		</div>
	);
}

export default App;