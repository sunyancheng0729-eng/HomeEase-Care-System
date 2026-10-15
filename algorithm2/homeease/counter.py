"""Debounced ordered cycle counting; interruptions invalidate an open cycle."""
from dataclasses import dataclass, field

PHASES = ["rest", "outbound", "hold", "return", "other"]


@dataclass
class CycleCounter:
    min_dwell_s: float = .067
    min_cycle_s: float = .6
    max_cycle_s: float = 30
    count: int = 0
    stage: int = -1
    candidate: str = "other"
    candidate_since: float = 0
    stable: str = "other"
    cycle_start: float | None = None
    events: list = field(default_factory=list)

    def interrupt(self):
        self.stage = -1
        self.cycle_start = None
        self.stable = self.candidate = "other"

    def update(self, phase, timestamp):
        if self.cycle_start is not None and timestamp - self.cycle_start > self.max_cycle_s:
            self.interrupt()
        if phase != self.candidate:
            self.candidate, self.candidate_since = phase, timestamp
            return False
        if timestamp - self.candidate_since + 1e-6 < self.min_dwell_s or phase == self.stable:
            return False
        self.stable = phase
        if phase == "other":
            self.interrupt()
            return False
        if phase == "rest":
            complete = self.stage == 3 and self.cycle_start is not None and timestamp - self.cycle_start >= self.min_cycle_s
            if complete:
                self.count += 1
                self.events.append({"rep_index": self.count, "start_s": self.cycle_start, "end_s": timestamp,
                                    "duration_s": timestamp - self.cycle_start})
            self.stage, self.cycle_start = 0, timestamp
            return complete
        expected = {"outbound": (0, 1), "hold": (1, 2), "return": (2, 3)}
        before, after = expected[phase]
        if self.stage == before:
            self.stage = after
        elif self.stage < after:
            self.stage, self.cycle_start = -1, None
        return False

    def update_probabilities(self, probabilities, timestamp, threshold=.35, infer_return=False):
        """Ordered posterior evidence can preserve short peaks below argmax."""
        if probabilities[4] >= .5:
            return self.update("other", timestamp)
        next_stage = 0 if self.stage < 0 else (self.stage + 1) % 4
        if infer_return and self.stage == 2 and probabilities[0] >= .6:
            # A visible return to baseline after a confirmed peak supplies return
            # evidence even if the classifier misses the brief transition phase.
            return self.update("return", timestamp)
        required = max(.5, threshold) if next_stage == 0 else threshold
        if probabilities[next_stage] >= required:
            return self.update(PHASES[next_stage], timestamp)
        # Preserve an already confirmed phase while waiting for the next one.
        current = self.stage if self.stage >= 0 else 4
        return self.update(PHASES[current], timestamp)
