"""
Which detector numbers were in the camera's view together.

The detector numbers each person it tracks. When it loses someone for a
moment it may give them a new number, and the backend maps the new number
back to the person it already holds (queue_service.remap_track_ids). Without
a face to compare, that mapping went on position alone: anyone within 180 px
of a person who had just dropped out of view.

That is exactly what happens when a third person steps in front of someone
already in line. The person behind is hidden, the detector drops their box,
and the newcomer standing where they were was taken for them. The newcomer
then counted as someone who already had a number: they were never treated
as new, and the detector, told that person was settled, stopped looking at
their face. On 24 Nov 2026 a student who had asked for a number stood in
view for 58 seconds before getting one.

Two numbers seen in the same update are two different people, so one can
never be the other's new number. A real renumbering looks different: the
old number disappears and the new one appears after it. Once the newcomer's
face has been seen not to match the person, that pair is ruled out as well.

Pure bookkeeping, no camera or database, so it can be tested directly.
"""


def similar_size(a, b) -> bool:
    """Neither box half as big again as the other."""
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return 2 / 3 <= area_a / area_b <= 3 / 2


class Sightings:
    def __init__(self, forget_after: int = 600):
        self.update = 0
        self.forget_after = forget_after
        self.first: dict = {}      # detector number -> update it was first seen in
        self.last: dict = {}       # detector number -> update it was last seen in
        self.seen_as: dict = {}    # tracker number -> {detector number: last update}
        self.not_same: set = set()  # (detector number, tracker number) whose faces differ
        self.face_on: dict = {}     # (detector number, tracker number) -> how many
                                    # face reads on that box were taken as theirs

    def begin(self, detector_ids) -> None:
        """A new update from the detector, with the numbers in it."""
        self.update += 1
        for d in detector_ids:
            self.first.setdefault(d, self.update)
            self.last[d] = self.update
        if self.update % 50 == 0:
            self._forget()

    def saw(self, detector_id, tracker_id) -> None:
        """In this update, the tracker's person `tracker_id` was the box the
        detector numbered `detector_id`."""
        self.seen_as.setdefault(tracker_id, {})[detector_id] = self.update

    def faces_differ(self, detector_id, tracker_id) -> None:
        self.not_same.add((detector_id, tracker_id))

    def face_taken(self, detector_id, tracker_id) -> None:
        """A face read on the box `detector_id` was stored as the face of
        the tracker's person `tracker_id`."""
        key = (detector_id, tracker_id)
        self.face_on[key] = self.face_on.get(key, 0) + 1

    def face_confirmed(self, detector_id, tracker_id) -> bool:
        """Whether the box `detector_id`, mapped to the tracker's person
        `tracker_id`, has shown that person's face twice. A box mapped by
        position alone has not: it may be someone who stepped in front of
        them. Nor has one mapped on a single read, which may have caught the
        face of the person hidden behind."""
        return self.face_on.get((detector_id, tracker_id), 0) >= 2

    def seen_together(self, detector_id, tracker_id) -> bool:
        """Whether the tracker's person `tracker_id` was seen under another
        number after the box `detector_id` came into view: two people."""
        first = self.first.get(detector_id, self.update)
        return any(at >= first for d, at in self.seen_as.get(tracker_id, {}).items()
                   if d != detector_id)

    def could_be(self, detector_id, tracker_id) -> bool:
        """Whether the box numbered `detector_id` may be the tracker's person
        `tracker_id` under a new number: not if the two were seen together,
        nor if a face on the box was seen not to match the face held for
        that person."""
        if (detector_id, tracker_id) in self.not_same:
            return False
        return not self.seen_together(detector_id, tracker_id)

    def held_as_someone_else(self, detector_id) -> bool:
        """Whether the box `detector_id` has been taken for a person the
        tracker holds under another number: then it is not a newcomer."""
        return any(tid != detector_id and detector_id in numbers
                   for tid, numbers in self.seen_as.items())

    def stepped_in_as_they_vanished(self, detector_id, tracker_id, window: int = 10) -> bool:
        """Whether the box `detector_id` came into view in the very update
        after the tracker's person `tracker_id` was last seen, and is still
        new (under `window` updates old). Either the detector renumbered that
        person, or someone stepped in front of them and hid them."""
        first = self.first.get(detector_id)
        seen = self.seen_as.get(tracker_id)
        if first is None or not seen:
            return False
        return first <= max(seen.values()) + 1 and self.update - first < window

    def _forget(self) -> None:
        cutoff = self.update - self.forget_after
        gone = {d for d, at in self.last.items() if at < cutoff}
        for d in gone:
            self.first.pop(d, None)
            self.last.pop(d, None)
        for tid in list(self.seen_as):
            numbers = {d: at for d, at in self.seen_as[tid].items() if at >= cutoff}
            if numbers:
                self.seen_as[tid] = numbers
            else:
                del self.seen_as[tid]
        self.not_same = {(d, t) for d, t in self.not_same if d not in gone}
        self.face_on = {k: n for k, n in self.face_on.items() if k[0] not in gone}
