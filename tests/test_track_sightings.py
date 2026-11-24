"""
Which detector numbers were in view together (services/track_sightings.py):
the rule that stops a newcomer being taken for someone they stood beside.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.track_sightings import Sightings  # noqa: E402


def see(s, *pairs):
    """One update: (detector number, tracker number) for each box in it."""
    s.begin([d for d, _ in pairs])
    for d, t in pairs:
        s.saw(d, t)


def test_two_numbers_seen_together_are_two_people():
    s = Sightings()
    see(s, (10, 10), (12, 12))      # student 1 and the newcomer, side by side
    see(s, (12, 12))                # student 1 hidden behind the newcomer
    assert not s.could_be(12, 10)


def test_a_real_renumbering_can_be_mapped_back():
    s = Sightings()
    see(s, (10, 10))
    see(s)                          # lost for a moment
    see(s, (70, 70))                # back under a new number
    assert s.could_be(70, 10)


def test_still_possible_once_mapped():
    s = Sightings()
    see(s, (10, 10))
    see(s, (70, 10))                # mapped back to the person
    see(s, (70, 10))
    assert s.could_be(70, 10), "being seen under the new number itself is no objection"


def test_a_face_that_did_not_match_rules_the_pair_out():
    s = Sightings()
    see(s, (10, 10))
    see(s, (70, 70))
    s.faces_differ(70, 10)
    assert not s.could_be(70, 10)


def test_a_renumbered_box_counts_as_that_person_after_two_of_their_face_reads():
    s = Sightings()
    assert not s.face_confirmed(70, 10), "mapped by position alone"
    s.face_taken(70, 10)
    assert not s.face_confirmed(70, 10), "one read may be the face of someone hidden behind"
    s.face_taken(70, 10)
    assert s.face_confirmed(70, 10)


def test_long_gone_numbers_are_forgotten():
    s = Sightings(forget_after=100)
    see(s, (10, 10), (12, 12))
    for _ in range(200):
        see(s, (99, 99))
    assert 12 not in s.first and 10 not in s.seen_as
    assert 99 in s.first
