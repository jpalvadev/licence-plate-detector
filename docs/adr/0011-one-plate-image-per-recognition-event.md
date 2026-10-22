# One plate image artifact per logged event

The Events log gives a reviewer numbers, not evidence: when a car's plate gets
recognized, they want the photo that justifies the row — especially for the
`PATENTE_ALFANUM` rows where Argentina/Paraguay/Peru 3+3 collisions need eyes.
The tracker already accumulates every sighting per plate track, and the pipeline
holds the plate crop exactly where it records a sighting, so retention is cheap
(a few tens of KB per open track, bounded by the miss tolerance). The one real
ambiguity: a track's highest-confidence sighting is **not** necessarily the
consensus winner's reading (a higher-confidence sighting can read a different
string, and ties break toward it), so a photo of the track-best crop could show
a plate contradicting the logged reading — the photo must be chosen among the
sightings that voted for the winner text.

Decision: every logged Events log row writes exactly one **Plate image artifact**
into the `MEDIA_DIR` folder (env knob, default `media/`, gitignored): one JPG
named `{plate}_{raw-conf}_{sanitized-timestamp}.jpg` — the raw 4-decimal
confidence and the row's first-sighting ISO timestamp with `:` replaced by
`-` (Windows forbids colons in filenames), so each photo pairs **1:1 with its
CSV row by name**. The crop is the best-confidence sighting among the winner's
votes; JPEG quality is a hardcoded 95 in `alpr/media.py` (evidence, one-line
re-anchor). Unvalidated consensus and dedup-suppressed closures write nothing —
the ADR-0009 "no row" rule extends verbatim: **no row, no artifact**.

Consequences: photos and rows pair by filename, so review can mean "row 412
looks wrong, here's the crop" — the raw conf in the name is model-scale, not
the display clarity % (the CSV keeps the raw value; the display seam re-anchors
independently). Folder growth is ~one small JPG per event until pruned
manually, same lifecycle policy as `events.csv`. Crops are retained per
sighting in memory (ADR-0006's sighting bookkeeping already keeps the rest).
`media_dir=None` disables artifacts for consumers that want rows only.

Rejected: one photo per sighting (duplicate spam at 3 FPS); the track's
highest-confidence crop regardless of text (photo could contradict the logged
reading); re-cropping from the drop-oldest frame buffer at event close
(the frames may already be gone by then); PNG (larger, no benefit for camera
grain).
