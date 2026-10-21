# Motorcycle plates are out of the domain

The original format table (ADR-0003) validated both Mercosur arrangements — car `AB 123 CD` and motorcycle `A 123 BCD` — and the implementation shipped that way (`MOTORCYCLE_RE`, `STATUS_MOTORCYCLE`, dashboard `.moto` styling, README table, tests). The operator stated the site's requirement outright: motorcycles are of no interest **in either era**, so they are removed from the domain rather than carried as dead weight.

Consequences: validation proves only `CAR`; the Events log's `vehicle_class` column takes only `CAR`/empty and the `status` column only `VALIDATED_CAR`, `VALIDATED_LEGACY` (see ADR-0008) or `UNVALIDATED`. An Argentine legacy motorcycle plate (`123 ABC`, digits first) can only ever land as an `UNVALIDATED` row — acceptable: the domain does not make claims about motorcycles, and unvalidated rows stay reviewable against the recording. The dashboard drops the blue moto style.

Rejected: keeping the motorcycle format "unused but harmless" (the operator's site decision makes it domain noise, and every test/table/status references left behind are maintenance liability); treating motorcycles as a third format-set mode (a mode nobody wants — the format set stays two-valued).

> Note: superseded in part by ADR-0009 — motorcycles stay out of the domain, but the `vehicle_class` column and `UNVALIDATED` status described in Consequences no longer exist.
