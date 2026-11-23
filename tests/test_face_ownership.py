"""
Whose face is it? With two people in view, a person's box often contains the
neighbour's face too. The detector must use the face that belongs to the box,
not the clearest one: on 23 Nov 2026 a track carried another student's face
38 times in 85 seconds, and the person on it, who had asked for a number,
never got one. No camera or face model needed.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.face_service import choose_own_face  # noqa: E402

# Two students side by side; their boxes overlap in the middle of the frame.
LEFT = (0, 100, 600, 700)        # student B, who asked for a number
RIGHT = (400, 80, 1000, 700)     # student A, already in line
B_FACE = (220, 140, 360, 300)    # centre x 290: middle of B's box
A_FACE = (490, 120, 630, 280)    # centre x 560: inside B's box, but near its edge


def test_one_face_in_the_box_is_that_persons():
    assert choose_own_face([B_FACE], LEFT) == 0


def test_a_neighbours_face_in_the_box_is_not_taken_for_this_persons():
    # Listed first, as the clearer face would have been picked before.
    assert choose_own_face([A_FACE, B_FACE], LEFT, [RIGHT]) == 1


def test_only_the_neighbours_face_visible_gives_no_face_rather_than_a_wrong_one():
    # B is looking down; only A's face is found in B's box.
    assert choose_own_face([A_FACE], LEFT, [RIGHT]) is None


def test_the_neighbour_still_gets_their_own_face():
    assert choose_own_face([A_FACE], RIGHT, [LEFT]) == 0


def test_with_nobody_else_in_view_the_most_central_face_is_chosen():
    poster = (20, 120, 120, 220)   # a face near the edge, e.g. a poster behind
    assert choose_own_face([poster, B_FACE], LEFT) == 1


def test_no_faces_gives_none():
    assert choose_own_face([], LEFT, [RIGHT]) is None
