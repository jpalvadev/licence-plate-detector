# Automatic License Plate Recognition (ALPR)

Argentine Mercosur-plate recognition from a single RTSP camera on a CPU-only server: plates are located, read, validated against the two Argentine formats, and deduplicated into events that are displayed and exported.

## Language

**Mercosur plate**:
A seven-character Argentine identification plate. The registration domain is four letters and three digits; the arrangement on the physical plate differs by vehicle class.
_Avoid_: license plate, chapa

**Car plate**:
A Mercosur plate for an automotor, arranged `AA 123 BB` (two letters, three digits, two letters).

**Motorcycle plate**:
A Mercosur plate for a motovehículo, arranged `A 123 BCD` (one letter, three digits, three letters).

**Vehicle class**:
The car-or-motorcycle category of a plate, derived from which format its string matches. The class is a property of the plate string itself, never of the camera or the vehicle pixels.
_Avoid_: type, kind

**Sighting**:
One observation of a plate on one detection pass: a bounding box, a raw reading, and a confidence. A sighting is an occurrence, not a record.
_Avoid_: detection, hit

**Raw reading**:
The text the OCR produces for a sighting, before normalization — may contain spaces, separators, or wrong characters.
_Avoid_: text, output

**Validated plate**:
A raw reading after normalization (uppercased, separators stripped) that matches the car or motorcycle format exactly. Only validated plates become events.
_Avoid_: clean text, valid string

**Unvalidated reading**:
A raw reading that does not match either format after normalization. Its sightings still form a deduplicated event (keyed by normalized raw text, carrying no Vehicle class), drawn as a grey box and logged with status `UNVALIDATED` so it can be reviewed against the recording later.
_Avoid_: failed OCR, garbage

**Recognition event**:
One occurrence of a plate: the same validated plate string sighted again within the dedup window extends the existing event (its confidence may improve) rather than creating a new one. Reappearance after the window starts a new event.
_Avoid_: detection, log entry, record

**Events log**:
The append-only CSV record of Recognition events — one row written the moment an event is created (first-sighting timestamp, status `VALIDATED_CAR` / `VALIDATED_MOTORCYCLE` / `UNVALIDATED`, confidence). Survives restarts; served to the frontend on demand.
_Avoid_: event buffer, history file

**Dedup window**:
The rolling time interval (default 10 s) inside which the same validated plate string maps to a single Recognition event.
_Avoid_: cooldown, TTL

**Detection pass**:
A frame on which plate detection and OCR actually run, sampled at the configured detection rate (default 3 FPS) rather than every source frame.
_Avoid_: processed frame

**Render frame**:
A frame of the displayed output stream (full source ~10 FPS). Its overlays come from the last detection pass; boxes persist between passes.
_Avoid_: displayed frame, output frame

**ROI**:
A normalized rectangle on the source frame; sightings whose box center falls outside it are discarded. Default is the full frame.
_Avoid_: crop, mask
