"""
Invalid read generator for Tempest robustness training.

Generates reads with segment-level architectural errors for training
models that are robust to common sequencing failures.

Part of: tempest/data module
"""

import numpy as np
import random
from typing import List, Dict, Optional
import logging

from tempest.data.simulator import SimulatedRead

logger = logging.getLogger(__name__)


class InvalidReadGenerator:
    """
    Generate invalid reads with segment-level architectural errors.
    
    Supports:
    - segment_loss: Missing UMI, ACC, or barcode
    - segment_duplication: Repeated segments (PCR artifacts)
    - truncation: Incomplete reads
    - chimeric: Mixed segments from different reads
    - scrambled: Randomized segment order
    """

    # ------------------------------------------------------------------ #
    # Initialization & config handling
    # ------------------------------------------------------------------ #
    def __init__(self, config=None):
        """Initialize invalid read generator."""
        self.config = config
        self.error_probabilities = self._init_error_probabilities(config)
        logger.info(f"Initialized InvalidReadGenerator with error probabilities: "
                    f"{self.error_probabilities}")

    @staticmethod
    def _default_probabilities() -> Dict[str, float]:
        """Return default error probabilities."""
        return {
            "segment_loss": 0.3,
            "segment_duplication": 0.3,
            "truncation": 0.2,
            "chimeric": 0.1,
            "scrambled": 0.1,
        }

    def _init_error_probabilities(self, config) -> Dict[str, float]:
        """Initialize error probabilities from config or defaults."""
        if not config:
            probs = self._default_probabilities()
        elif hasattr(config, "hybrid") and config.hybrid:
            hybrid = config.hybrid
            probs = {
                "segment_loss": getattr(hybrid, "segment_loss_prob", 0.3),
                "segment_duplication": getattr(hybrid, "segment_dup_prob", 0.3),
                "truncation": getattr(hybrid, "truncation_prob", 0.2),
                "chimeric": getattr(hybrid, "chimeric_prob", 0.1),
                "scrambled": getattr(hybrid, "scrambled_prob", 0.1),
            }
        elif isinstance(config, dict) and "hybrid" in config:
            hybrid = config["hybrid"]
            probs = {
                "segment_loss": hybrid.get("segment_loss_prob", 0.3),
                "segment_duplication": hybrid.get("segment_dup_prob", 0.3),
                "truncation": hybrid.get("truncation_prob", 0.2),
                "chimeric": hybrid.get("chimeric_prob", 0.1),
                "scrambled": hybrid.get("scrambled_prob", 0.1),
            }
        else:
            probs = self._default_probabilities()

        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()} if total > 0 else probs

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    def _pychoice(self, items):
        """Return one random element from a list, preserving dicts."""
        if not items:
            return None
        choice = random.choice(items)
        return choice.copy() if isinstance(choice, dict) else choice

    @staticmethod
    def _clone_metadata(read: SimulatedRead, **updates) -> Dict:
        """Copy metadata and apply updates."""
        meta = dict(read.metadata) if read.metadata else {}
        meta.update(updates)
        return meta

    @staticmethod
    def _reassemble(read: SimulatedRead, segments: List[Dict]) -> tuple[str, list[str], dict]:
        """Rebuild sequence, labels, and regions from given segment definitions."""
        seq, labels, regions, pos = "", [], {}, 0
        for seg in segments:
            s, e = seg["start"], seg["end"]
            seg_seq = read.sequence[s:e]
            seg_labels = read.labels[s:e]
            seq += seg_seq
            labels.extend(seg_labels)
            regions.setdefault(seg["type"], []).append((pos, pos + len(seg_seq)))
            pos += len(seg_seq)
        return seq, labels, regions

    # ------------------------------------------------------------------ #
    # Main API
    # ------------------------------------------------------------------ #
    def generate_invalid_read(self, valid_read: SimulatedRead,
                              error_type: Optional[str] = None) -> SimulatedRead:
        """Generate an invalid read from a valid one."""
        if error_type is None:
            # Weighted choice uses numpy since random.choice lacks probability argument
            error_type = np.random.choice(
                list(self.error_probabilities.keys()),
                p=list(self.error_probabilities.values())
            )

        handlers = {
            "segment_loss": self._apply_segment_loss,
            "segment_duplication": self._apply_segment_duplication,
            "truncation": self._apply_truncation,
            "chimeric": self._apply_chimeric,
            "scrambled": self._apply_scrambled,
        }
        return handlers.get(error_type, lambda r: r)(valid_read)

    def generate_batch(self, valid_reads: List[SimulatedRead],
                       invalid_ratio: float = 0.1) -> List[SimulatedRead]:
        """Generate a batch of invalid reads from valid ones."""
        if not valid_reads:
            return []

        # Always produce at least one invalid read if ratio > 0
        n_invalid = max(1, int(len(valid_reads) * invalid_ratio)) if invalid_ratio > 0 else 0
        if n_invalid == 0:
            return valid_reads

        # Use a set for efficient membership check (robust for single-element arrays)
        choice = np.random.choice(len(valid_reads), n_invalid, replace=False)
        indices = set(np.atleast_1d(choice).tolist())

        return [
            self.generate_invalid_read(r) if i in indices else r
            for i, r in enumerate(valid_reads)
        ]

    # ------------------------------------------------------------------ #
    # Individual corruption strategies
    # ------------------------------------------------------------------ #
    def _apply_segment_loss(self, read: SimulatedRead) -> SimulatedRead:
        """Remove a random segment (e.g., missing barcode or UMI)."""
        segments = self._extract_segments(read)
        removable = [s for s in segments
                     if "ADAPTER" not in s["type"] and "INSERT" not in s["type"]]
        if not removable:
            return read

        to_remove = self._pychoice(removable)
        if not isinstance(to_remove, dict):
            return read

        new_seq = read.sequence[:to_remove["start"]] + read.sequence[to_remove["end"]:]
        new_labels = read.labels[:to_remove["start"]] + read.labels[to_remove["end"]:]
        removed_len = to_remove["end"] - to_remove["start"]

        new_regions = {}
        for label, regions in read.label_regions.items():
            if label == to_remove["type"]:
                continue
            adjusted = []
            for start, end in regions:
                if end <= to_remove["start"]:
                    adjusted.append((start, end))
                elif start >= to_remove["end"]:
                    adjusted.append((start - removed_len, end - removed_len))
                elif start < to_remove["start"] and end > to_remove["end"]:
                    adjusted.extend([
                        (start, to_remove["start"]),
                        (to_remove["start"], end - removed_len)
                    ])
            if adjusted:
                new_regions[label] = adjusted

        meta = self._clone_metadata(read,
                                    error_type="segment_loss",
                                    removed_segment=to_remove["type"])
        return SimulatedRead(new_seq, new_labels, new_regions, meta)

    def _apply_segment_duplication(self, read: SimulatedRead) -> SimulatedRead:
        """Duplicate a random segment (PCR-like artifact)."""
        segments = self._extract_segments(read)
        duplicatable = [s for s in segments if s["type"] in ["UMI", "BARCODE", "ACC"]] \
                       or [s for s in segments if s["type"] != "INSERT"]
        if not duplicatable:
            return read

        to_dup = self._pychoice(duplicatable)
        seg_seq = read.sequence[to_dup["start"]:to_dup["end"]]
        seg_labels = read.labels[to_dup["start"]:to_dup["end"]]
        dup_len = to_dup["end"] - to_dup["start"]

        new_seq = read.sequence[:to_dup["end"]] + seg_seq + read.sequence[to_dup["end"]:]
        new_labels = read.labels[:to_dup["end"]] + seg_labels + read.labels[to_dup["end"]:]

        new_regions = {}
        for label, regions in read.label_regions.items():
            adjusted = []
            for start, end in regions:
                if label == to_dup["type"] and start == to_dup["start"]:
                    adjusted.extend([(start, end), (end, end + dup_len)])
                elif start >= to_dup["end"]:
                    adjusted.append((start + dup_len, end + dup_len))
                else:
                    adjusted.append((start, end))
            if adjusted:
                new_regions[label] = adjusted

        meta = self._clone_metadata(read,
                                    error_type="segment_duplication",
                                    duplicated_segment=to_dup["type"])
        return SimulatedRead(new_seq, new_labels, new_regions, meta)

    def _apply_truncation(self, read: SimulatedRead) -> SimulatedRead:
        """Truncate the read to simulate incomplete sequencing."""
        min_len = max(10, len(read.sequence) // 2)
        max_len = int(len(read.sequence) * 0.9)
        cut = random.randint(min_len, max_len) if min_len < max_len else min_len
        meta = self._clone_metadata(read, error_type="truncation", truncation_point=cut)
        new_regions = self._update_regions_for_truncation(read.label_regions, cut)
        return SimulatedRead(read.sequence[:cut], read.labels[:cut], new_regions, meta)

    def _apply_chimeric(self, read: SimulatedRead,
                        other_read: Optional[SimulatedRead] = None) -> SimulatedRead:
        """Scramble middle segments while keeping adapters intact."""
        segments = self._extract_segments(read)
        if len(segments) < 4:
            return read
        middle = segments[1:-1]
        random.shuffle(middle)
        ordered = [segments[0]] + middle + [segments[-1]]
        seq, labels, regs = self._reassemble(read, ordered)
        meta = self._clone_metadata(read, error_type="chimeric")
        return SimulatedRead(seq, labels, regs, meta)

    def _apply_scrambled(self, read: SimulatedRead) -> SimulatedRead:
        """Completely randomize segment order."""
        segments = self._extract_segments(read)
        if len(segments) <= 1:
            return read
        random.shuffle(segments)
        seq, labels, regs = self._reassemble(read, segments)
        meta = self._clone_metadata(read, error_type="scrambled")
        return SimulatedRead(seq, labels, regs, meta)

    # ------------------------------------------------------------------ #
    # Segment utilities
    # ------------------------------------------------------------------ #
    def _extract_segments(self, read: SimulatedRead) -> List[Dict]:
        """Extract segment boundaries and labels."""
        if not read.labels:
            return []
        segments, start, current = [], 0, read.labels[0]
        for i, label in enumerate(read.labels[1:], 1):
            if label != current:
                segments.append({"type": current, "start": start, "end": i})
                current, start = label, i
        segments.append({"type": current, "start": start, "end": len(read.labels)})
        return segments

    @staticmethod
    def _update_regions_for_truncation(regions: Dict, cut_point: int) -> Dict:
        """Trim region definitions after truncation."""
        out = {}
        for label, lst in regions.items():
            kept = [(s, min(e, cut_point)) for s, e in lst if s < cut_point]
            if kept:
                out[label] = kept
        return out