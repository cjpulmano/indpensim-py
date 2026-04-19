"""UNS-style topic builder + tag mapping for the learning-sim integration.

The learning-sim project uses topics of the form

    uns/{site}/{area}/{line}/{equipment}/{tag}             # process value
    uns/{site}/{area}/{line}/{equipment}/_state            # unit state JSON
    uns/{site}/{area}/{line}/{equipment}/_phase_start      # phase transition

published every 2 wall-seconds. This module turns one ``Sample`` into a
list of (topic, payload_json_bytes) tuples ready to hand to an MQTT client.

Indpensim publishes under a configurable equipment id (default
``bioreactor-real``) so it can run side-by-side with learning-sim's toy
``bioreactor-001`` for direct comparison in Grafana.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from indpensim.streaming.sample import Sample


# ---------------------------------------------------------------------------
# Tag mapping: indpensim internal name → (UNS tag name, unit, conversion)
# ---------------------------------------------------------------------------

def _identity(x: float) -> float:
    return x


def _kelvin_to_celsius(K: float) -> float:
    return K - 273.15


@dataclass(frozen=True)
class _TagMap:
    uns_name: str
    unit: str
    convert: callable = _identity


# Process-tag mapping. Subset of the 33 states + derived channels we want
# to publish — no point streaming the 10 vacuole bins (n0..n9) to a SCADA
# system, they're internal model state.
_STATE_TAG_MAP: dict[str, _TagMap] = {
    "T":           _TagMap("temperature",       "degC",  _kelvin_to_celsius),
    "pH":          _TagMap("ph",                "-"),
    "DO2":         _TagMap("dissolved_o2",      "mg/L"),
    "O2":          _TagMap("offgas_o2",         "frac"),
    "CO2outgas":   _TagMap("offgas_co2",        "frac"),
    "S":           _TagMap("glucose",           "g/L"),
    "P":           _TagMap("penicillin_titre",  "g/L"),
    "V":           _TagMap("volume",            "L"),
    "Wt":          _TagMap("weight",            "kg"),
    "Viscosity":   _TagMap("viscosity",         "cP"),
    "X":           _TagMap("biomass",           "g/L"),
    "PAA":         _TagMap("paa_concentration", "mg/L"),
    "NH3":         _TagMap("nh3_concentration", "mg/L"),
    "Culture_age": _TagMap("culture_age",       "h"),
    "OUR":         _TagMap("our",               "g/h"),
    "CER":         _TagMap("cer",               "g/h"),
    "mu_X_calc":   _TagMap("mu_biomass_growth", "1/h"),
    "mu_P_calc":   _TagMap("mu_pen_growth",     "1/h"),
}

_CONTROL_TAG_MAP: dict[str, _TagMap] = {
    "Fc":       _TagMap("coolant_flow",   "L/h"),
    "Fh":       _TagMap("heating_flow",   "L/h"),
    "Fb":       _TagMap("base_flow",      "L/h"),
    "Fa":       _TagMap("acid_flow",      "L/h"),
    "Fs":       _TagMap("substrate_flow", "L/h"),
    "Fpaa":     _TagMap("paa_flow",       "L/h"),
    "Fg":       _TagMap("aeration_rate",  "L/h"),
    "Foil":     _TagMap("oil_flow",       "L/h"),
    "Fw":       _TagMap("water_flow",     "L/h"),
    "Fremoved": _TagMap("discharge_flow", "L/h"),
    "RPM":      _TagMap("agitation_rpm",  "rpm"),
    "pressure": _TagMap("vessel_pressure","bar"),
}


@dataclass(frozen=True)
class UnsConfig:
    """Static identity of this simulator instance in the UNS namespace."""
    site: str = "plant1"
    area: str = "fermentation"
    line: str = "line1"
    equipment: str = "bioreactor-real"

    def topic(self, suffix: str) -> str:
        return f"uns/{self.site}/{self.area}/{self.line}/{self.equipment}/{suffix}"


def _now_iso() -> str:
    """ISO-8601 UTC timestamp matching the format learning-sim expects."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_messages(sample: Sample, cfg: UnsConfig,
                   include_raman: bool = True,
                   include_offline: bool = True,
                   ) -> list[tuple[str, bytes]]:
    """Turn one Sample into a list of (topic, payload_json_bytes) tuples.

    Process tags emitted every call. Raman/offline emitted only when the
    Sample carries those fields (controlled by ``StreamConfig`` upstream).

    Payload schema for each tag matches what learning-sim's bridge expects:
        { "value": <float>, "unit": <str>, "ts": <iso-8601>, "k": <int> }
    """
    ts = _now_iso()
    msgs: list[tuple[str, bytes]] = []

    for src_name, tagmap in _STATE_TAG_MAP.items():
        raw = sample.state.get(src_name)
        if raw is None:
            continue
        payload = {
            "value": float(tagmap.convert(raw)),
            "unit": tagmap.unit,
            "ts": ts,
            "k": sample.k,
        }
        msgs.append((cfg.topic(tagmap.uns_name),
                     json.dumps(payload).encode("utf-8")))

    for src_name, tagmap in _CONTROL_TAG_MAP.items():
        raw = sample.controls.get(src_name)
        if raw is None:
            continue
        payload = {
            "value": float(tagmap.convert(raw)),
            "unit": tagmap.unit,
            "ts": ts,
            "k": sample.k,
        }
        msgs.append((cfg.topic(tagmap.uns_name),
                     json.dumps(payload).encode("utf-8")))

    # Raman: one big payload (large, slow cadence).
    if include_raman and sample.raman is not None:
        payload = {
            "intensity": sample.raman,
            "ts": ts,
            "k": sample.k,
            "n_bins": len(sample.raman),
        }
        msgs.append((cfg.topic("raman_spectrum"),
                     json.dumps(payload).encode("utf-8")))

    # Lab/offline: one message per analyte.
    if include_offline and sample.offline is not None:
        for analyte, value in sample.offline.items():
            payload = {
                "value": float(value),
                "unit": "g/L" if analyte in ("P", "X") else "mg/L",
                "ts": ts,
                "k": sample.k,
                "source": "offline_lab",
            }
            msgs.append((cfg.topic(f"lab_{analyte}"),
                         json.dumps(payload).encode("utf-8")))

    return msgs


def build_state_message(sample: Sample, cfg: UnsConfig,
                         phase: str = "FERMENT",
                         batch_id: int | None = None,
                         elapsed_h: float | None = None,
                         ) -> tuple[str, bytes]:
    """Build the unit-state heartbeat (matches learning-sim's _state schema)."""
    payload = {
        "phase": phase,
        "batch_id": batch_id,
        "elapsed_h": elapsed_h if elapsed_h is not None else sample.sim_time_h,
        "k": sample.k,
        "sim_time_h": sample.sim_time_h,
        "ts": _now_iso(),
        "running": True,
    }
    return cfg.topic("_state"), json.dumps(payload).encode("utf-8")


def build_phase_start_message(cfg: UnsConfig, *, phase: str, batch_id: int,
                              phase_index: int) -> tuple[str, bytes]:
    """Build a phase-transition event (written to learning-sim's batch_events)."""
    payload = {
        "phase": phase,
        "batch_id": batch_id,
        "phase_index": phase_index,
        "ts": _now_iso(),
    }
    return cfg.topic("_phase_start"), json.dumps(payload).encode("utf-8")
