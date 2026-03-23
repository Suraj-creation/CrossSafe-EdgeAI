"""
Stage 3 — Multi-Object Tracking (ByteTrack) & Speed Estimation.

ByteTrack is purely IoU-based — no re-ID CNN needed.
~5ms on Pi 4 vs ~50ms for DeepSORT.
Speed is estimated from bbox displacement across frames using
a pixel-to-metre calibration factor set at installation.
"""

import numpy as np
from collections import defaultdict, deque


class VehicleTracker:
    def __init__(
        self,
        pixels_per_metre: float = 47.0,
        camera_fps: float = 15.0,
        speed_threshold_kmph: float = 5.0,
        cooldown_seconds: float = 60.0,
        max_history: int = 15,
    ):
        self.pixels_per_metre = pixels_per_metre
        self.camera_fps = camera_fps
        self.speed_threshold = speed_threshold_kmph
        self.cooldown_seconds = cooldown_seconds

        self.track_positions: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.track_speeds: dict[int, float] = {}
        self.challan_timestamps: dict[int, float] = {}

    def update(self, track_id: int, center: tuple[int, int]) -> float:
        """
        Record position for a tracked vehicle and return estimated speed (km/h).
        Returns 0.0 if insufficient history for estimation.
        """
        self.track_positions[track_id].append(center)

        if len(self.track_positions[track_id]) < 4:
            return 0.0

        pts = list(self.track_positions[track_id])[-4:]
        dists = [
            np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts))
        ]
        avg_px_per_frame = float(np.mean(dists))

        metres_per_frame = avg_px_per_frame / self.pixels_per_metre
        metres_per_second = metres_per_frame * self.camera_fps
        speed_kmph = round(metres_per_second * 3.6, 1)

        self.track_speeds[track_id] = speed_kmph
        return speed_kmph

    def is_moving_violation(self, track_id: int) -> bool:
        return self.track_speeds.get(track_id, 0.0) >= self.speed_threshold

    def is_in_cooldown(self, track_id: int, current_time: float) -> bool:
        if track_id not in self.challan_timestamps:
            return False
        elapsed = current_time - self.challan_timestamps[track_id]
        return elapsed < self.cooldown_seconds

    def record_challan(self, track_id: int, timestamp: float):
        self.challan_timestamps[track_id] = timestamp

    def get_speed(self, track_id: int) -> float:
        return self.track_speeds.get(track_id, 0.0)

    def cleanup_stale(self, active_ids: set[int]):
        """Remove state for tracks no longer visible."""
        stale = set(self.track_positions.keys()) - active_ids
        for tid in stale:
            self.track_positions.pop(tid, None)
            self.track_speeds.pop(tid, None)
