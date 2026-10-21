# Automatic License Plate Recognition (ALPR)

Argentine plate recognition from a single RTSP camera on a CPU-only server: plates are located, read, validated against a configured format set, and deduplicated into events that are displayed and exported.

## Language

**Mercosur plate**:
A seven-character Argentine identification plate of the current (Mercosur) era, arranged `AB 123 CD` (two letters, three digits, two letters).
_Avoid_: license plate, chapa

**Legacy plate**:
A six-character Argentine identification plate of the 1995–2016 era, arranged `ABC 123` (three letters, three digits) — superseded by Mercosur plates only for new registrations from April 2016, so they remain common on older vehicles. Whether legacy plates are in play at all is set by the format set, not by the camera. It carries no vehicle class in its characters (in that era car and cargo variants shared the same characters).
_Avoid_: old plate, old format

**Sighting**:
One observation of a plate on one detection pass: a bounding box, a raw reading, and a confidence. A sighting is an occurrence, not a record.
_Avoid_: detection, hit

**Raw reading**:
The text the OCR produces for a sighting, before normalization — may contain spaces, separators, or wrong characters.
_Avoid_: text, output

**Validated plate**:
A raw reading after normalization (uppercased, separators stripped) that matches one of the formats in the format set exactly. Only a track whose consensus reading is a validated plate becomes an event row (ADR-0009).
_Avoid_: clean text, valid string

**Unvalidated reading**:
A raw reading that matches no format in the format set after normalization. Its sightings join the same plate track as better readings of the same box; when a track's consensus comes out unvalidated the event logs no row at all (ADR-0009) — it stays visible only in the live render and in the camera recording.
_Avoid_: failed OCR, garbage

**Recognition event**:
One occurrence of a physical plate, bounded by a plate track rather than by a string: consecutive detections of the same box in motion belong to one event regardless of what the OCR read on each pass. An event with a validated consensus yields exactly one row in the Events log, carrying the plurality reading of its sightings (ties → higher confidence); an event whose consensus is unvalidated yields no row at all (ADR-0009). Reappearance of a plate after the event closed is a new event.
_Avoid_: detection, log entry, record, sighting

**Plate track**:
The spatial continuity of a plate across detection passes: a sighting whose box matches an open track's box from a previous pass extends that track; an unmatched sighting opens a new one. A track tolerates a few consecutive passes where the plate was not detected; exceeding that tolerance closes it, and so does stream loss. One closed track = one Recognition event.
_Avoid_: tracker (that's the mechanism), blob, detection chain

**Events log**:
The append-only CSV record of logged Recognition events — one row written when an event **closes**, carrying the event's consensus reading over all its sightings (not the first sighting's reading). Only events with a validated consensus are logged (ADR-0009). Survives restarts; served to the frontend on demand.
_Avoid_: event buffer, history file

**Dedup window**:
The rolling time interval (default 10 s) whose only remaining role is a guard: once a Recognition event has closed with a given plate string, sightings producing that exact string within the window add no new Events log row (they belong to the same occurrence or a split track of it).
_Avoid_: cooldown, TTL

**Detection pass**:
A frame on which plate detection and OCR actually run, sampled at the configured detection rate (default 3 FPS) rather than every source frame.
_Avoid_: processed frame

**Render frame**:
A frame of the displayed output stream (full source ~10 FPS). Its overlays come from the last detection pass; boxes persist between passes.
_Avoid_: displayed frame, output frame

**Format set**:
The startup choice of which plate formats are validatable: the Mercosur car format alone, or that plus the legacy car format (motorcycles are out of the domain, ADR-0007). A domain decision, not a code fork: the same detection, tracking and event machinery runs under either setting.

**ROI**:
A normalized rectangle on the source frame; sightings whose box center falls outside it are discarded. Default is the full frame.
_Avoid_: crop, mask

**Test rig**:
A simulated camera on the dev machine: `sample.mp4` (gitignored, real plates) pushed on an infinite loop through a local mediamtx over RTSP, and read by the app exactly as it reads a real camera — same ingest thread, same ffmpeg pipe. Exercises the RTSP path without the server.
_Avoid_: fake stream, demo, debug source

**Private runbook**:
`private.toml` at the repo root — the gitignored, machine-specific operational checklist (real endpoints, local commands, server paths). The public README carries only the generic, parameter-less version of the same procedure.
_Avoid_: config (that's `.env`), .env.example
