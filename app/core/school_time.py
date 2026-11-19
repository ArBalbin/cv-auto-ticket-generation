"""Times shown to people, on the school's clock.

The backend runs on Render, whose clock is UTC, so a time formatted straight
from datetime.now() read eight hours behind in Naga ("Joined 12:30 PM" for a
student who joined at 8:30 PM). Two kinds of time reach the displays:

- times this process takes itself (datetime.now(), naive, on the server's own
  clock: UTC on Render, local time on a laptop), and
- times the database stamps with NOW(), which are UTC whatever this process's
  clock is.

Both are converted to the school's offset before they are formatted. The
machine-readable ``*_iso`` fields and the CSV export keep their original values.
"""
from datetime import datetime, timedelta, timezone

from core.config import SCHOOL_UTC_OFFSET_HOURS

SCHOOL_TZ = timezone(timedelta(hours=SCHOOL_UTC_OFFSET_HOURS))

CLOCK = "%I:%M:%S %p"
CLOCK_FULL = "%b %d, %Y %I:%M:%S %p"


def on_school_clock(moment: datetime) -> datetime:
    """A time this process took (naive = the server's own clock)."""
    return moment.astimezone(SCHOOL_TZ)


def db_on_school_clock(moment: datetime) -> datetime:
    """A time the database stamped with NOW(), which is UTC."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(SCHOOL_TZ)


def clock(moment: datetime, fmt: str = CLOCK) -> str:
    """A server time as '08:30:31 PM' on the school's clock."""
    return on_school_clock(moment).strftime(fmt)


def db_clock(moment: datetime, fmt: str = CLOCK) -> str:
    """A database time as '08:30:31 PM' on the school's clock."""
    return db_on_school_clock(moment).strftime(fmt)
