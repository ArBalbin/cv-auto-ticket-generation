"""
Which people the detector turns into face signatures, and when.

The detector used to compute a face signature for every person in view on
every frame, one after another, before it could track the next frame. At
about 260 ms a face on the office laptop, three people cost almost a second
a frame. Tracking slowed to about one frame a second, so people who moved or
crossed were lost and came back as new tracks, and the extra signatures were
mostly thrown away, because the backend receives only a few updates a
second.

Face work now runs in its own worker on a budget:

- People the backend has not identified yet come first, the one tried the
  fewest times first, so someone who just walked in is looked at at once.
  After FACE_WARMUP tries, a person is tried less often, so someone who will
  never match (a walk-in, or someone facing away) cannot starve the rest.
  While a student who asked for a number is still waiting for one, the
  backend says so (faces_wanted) and nobody is slowed down.
- People the backend has already resolved (identified, or recognized but
  not queueing) are refreshed only every FACE_REFRESH_SECONDS, which keeps
  the backend's face checks on them current.
- Each signature is sent to the backend exactly once. A signature repeated
  in several updates would count as several sightings of the face.

The functions here are pure, so they can be tested without a camera.
"""
from dataclasses import dataclass, field


@dataclass
class FaceTrackState:
    first_seen: float
    tries: int = 0
    successes: int = 0
    last_try: float = 0.0
    last_success: float = 0.0


@dataclass
class FaceSchedule:
    per_round: int = 2
    retry_gap: float = 0.15
    slow_gap: float = 1.0
    warmup: int = 6
    refresh_seconds: float = 2.0
    forget_seconds: float = 10.0
    state: dict = field(default_factory=dict)

    def choose(self, track_ids, resolved: set, now: float, urgent: bool = False) -> list:
        """The tracks to compute a face signature for now, in order.

        `urgent` is the backend saying a student has asked for a number and
        not been given one (faces_wanted). Unidentified people are then
        retried at the quick pace however many tries they have had: a
        student who stood in view for a while before asking, or who came
        close only after several misses from afar, was otherwise looked at
        once a second. They still take turns, fewest tries first."""
        unresolved, refresh = [], []
        for tid in track_ids:
            s = self.state.get(tid)
            if s is None:
                s = self.state[tid] = FaceTrackState(first_seen=now)
            since_try = now - s.last_try
            if tid in resolved:
                if (now - s.last_success >= self.refresh_seconds
                        and since_try >= self.refresh_seconds):
                    refresh.append((s.last_success, tid))
                continue
            gap = self.retry_gap if (urgent or s.tries < self.warmup) else self.slow_gap
            if since_try >= gap:
                unresolved.append((s.tries, s.first_seen, tid))
        unresolved.sort()
        refresh.sort()
        ordered = [t for _, _, t in unresolved] + [t for _, t in refresh]
        return ordered[:self.per_round]

    def record(self, tid, ok: bool, now: float) -> None:
        s = self.state.setdefault(tid, FaceTrackState(first_seen=now))
        s.tries += 1
        s.last_try = now
        if ok:
            s.successes += 1
            s.last_success = now

    def forget_absent(self, present: set, now: float) -> None:
        """Drop tracks that have been out of view for forget_seconds."""
        for tid in list(self.state):
            s = self.state[tid]
            if tid not in present and now - max(s.last_try, s.first_seen) > self.forget_seconds:
                del self.state[tid]


def attach_fresh_embeddings(tracked_persons: list, fresh: dict) -> list:
    """Put each unsent signature on its person in an outgoing update.

    Removes what it attaches from `fresh`, so each signature goes out once;
    every other person goes out with no signature.
    """
    out = []
    for p in tracked_persons:
        q = dict(p)
        q["face_embedding"] = fresh.pop(p["track_id"], None)
        out.append(q)
    return out


def resolved_track_ids(reply: dict) -> set:
    """The tracks the backend no longer needs faces for, from its reply.

    Newer backends list them in face_resolved (identified students and
    recognized students who have not asked for a number). An older backend
    only lists numbered people in queue_state.active_queue.
    """
    if isinstance(reply.get("face_resolved"), list):
        return {int(t) for t in reply["face_resolved"]}
    state = reply.get("queue_state") or {}
    return {
        int(p["track_id"]) for p in state.get("active_queue", [])
        if p.get("track_id") is not None and p.get("queue_number") is not None
    }
