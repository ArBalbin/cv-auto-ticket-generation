import os
import io
import jwt
import secrets
import string
import threading
import time
import qrcode
from datetime import datetime, timezone, timedelta
from PIL import Image

from reportlab.lib.units import mm as mmUnit
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader

from core.config import JWT_SECRET_KEY, PORTAL_BASE_URL, TICKETS_OUTPUT_DIR
from services import object_storage_service


# CONFIGURATION
# In the real queue area, this module can be replaced or extended to print
# through a thermal printer. For the prototype/cloud demo, it generates a PDF
# ticket with the same queue number, QR code, and access token.
TICKET_WIDTH_MM  = 80
# 120mm, down from 175mm. The ticket now carries only what a student needs in
# their hand; the rows that were removed were either stale the moment the
# paper was cut (position, estimated wait, counters open — all of which keep
# changing while the student waits, which is exactly what the QR code is for)
# or internal plumbing they cannot act on ("JWT SECURED", token expiry).
TICKET_HEIGHT_MM = 120

INSTITUTION = "Naga College Foundation, Inc."

# Base URL for QR code — student scans and lands on their live status page.
# Set PORTAL_BASE_URL in .env for your server IP or domain.
# e.g. http://192.168.1.10:5000  or  https://queueflow.ncf.edu
JWT_ALGORITHM    = "HS256"
JWT_EXPIRY_HOURS = 4

# Plain black on white. Beyond looking cleaner, this is the only palette a
# thermal receipt printer can actually reproduce — the module docstring above
# names that as the real deployment path, and a dark-background ticket would
# be unprintable on one.
COLOR_INK    = HexColor("#000000")   # queue number, short code, student name
COLOR_MUTED  = HexColor("#666666")   # small labels
COLOR_RULE   = HexColor("#CCCCCC")   # hairline separators


# JWT TOKEN GENERATION

def generate_jwt_token(queue_number: int, service: str = "Enrollment Office") -> str:
    now    = datetime.now(timezone.utc)
    expiry = now + timedelta(hours=JWT_EXPIRY_HOURS)
    payload = {
        "sub" : str(queue_number),
        "svc" : service,
        "iat" : now,
        "exp" : expiry,
        "jti" : secrets.token_hex(16),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def generate_short_code() -> str:
    chars = (string.ascii_uppercase + string.digits).translate(
        str.maketrans('', '', '0O1I')
    )
    raw = ''.join(secrets.choice(chars) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"

# JWT TOKEN VALIDATION

def validate_jwt_token(token: str) -> dict | None:
    try:
        return jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={
                "require": ["sub", "exp", "iat", "jti"],
                "verify_sub": False,
            }
        )
    except jwt.ExpiredSignatureError:
        print("[JWT] Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"[JWT] Invalid token: {e}")
        return None


def validate_short_code(short_code: str, queue_number: int, db_pool) -> dict | None:
    try:
        record = get_ticket_record_by_short_code(short_code, queue_number, db_pool)
    except TicketLookupBusy:
        return None
    if not record or record.get("status") != "waiting":
        return None
    return record.get("jwt_payload")


# Every waiting student's app polls its ticket status every few seconds, and
# each poll used to take a database connection just to re-validate the same
# ticket. At a few hundred simultaneous students that exhausted the connection
# pool. A ticket that was valid and waiting a moment ago is still valid now, so
# it is remembered briefly; anything that changes a ticket's status calls
# invalidate_ticket_cache(), and the TTL bounds staleness for anything else.
_TICKET_CACHE_TTL_SECONDS = 10.0
# How long past expiry a cached ticket may still answer when the database is
# busy or unreachable, rather than telling a valid ticket-holder "invalid".
_TICKET_STALE_GRACE_SECONDS = 60.0
# A short code that does not exist is remembered briefly too, so a client
# polling a wrong code cannot make every poll a database read.
_TICKET_MISS_TTL_SECONDS = 3.0
# Longest a poll waits for another poll of the same ticket to finish reading.
_TICKET_LOCK_WAIT_SECONDS = 3.0
_ticket_cache: dict[tuple[str, int], tuple[float, dict | None]] = {}
_ticket_cache_lock = threading.Lock()
_ticket_key_locks: dict[tuple[str, int], threading.Lock] = {}
_MISSING = object()


class TicketLookupBusy(Exception):
    """The ticket could not be checked right now (database busy). Callers
    answer 503 "try again" rather than calling a valid ticket invalid."""


def invalidate_ticket_cache(queue_number: int | None = None) -> None:
    """Forget cached tickets: one queue number, or all of them."""
    with _ticket_cache_lock:
        if queue_number is None:
            _ticket_cache.clear()
            _ticket_key_locks.clear()
        else:
            for key in [k for k in _ticket_cache if k[1] == queue_number]:
                _ticket_cache.pop(key, None)


def _cached_ticket(key, now: float, allow_stale: bool = False):
    """The cached answer: a ticket dict, None for a known-missing ticket, or
    _MISSING when nothing usable is cached."""
    with _ticket_cache_lock:
        hit = _ticket_cache.get(key)
    if not hit:
        return _MISSING
    expires, row = hit
    if row is None:
        return None if expires > now else _MISSING
    if expires > now or (allow_stale and expires + _TICKET_STALE_GRACE_SECONDS > now):
        return dict(row)
    return _MISSING


def _ticket_key_lock(key) -> threading.Lock:
    with _ticket_cache_lock:
        lock = _ticket_key_locks.get(key)
        if lock is None:
            lock = _ticket_key_locks[key] = threading.Lock()
        return lock


def _fetch_ticket_row(short_code: str, queue_number: int, db_pool) -> dict | None:
    conn = cursor = None
    try:
        conn   = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, queue_number, short_code, jwt_token, status, "
            "created_at, served_at, expires_at "
            "FROM queue_records "
            "WHERE short_code = %s "
            "AND queue_number = %s "
            "AND (expires_at IS NULL OR expires_at > NOW()) "
            "ORDER BY created_at DESC, id DESC "
            "LIMIT 1",
            (short_code, queue_number)
        )
        return cursor.fetchone()
    finally:
        # Always hand the connection back; an exception used to leak it.
        for resource in (cursor, conn):
            try:
                if resource is not None:
                    resource.close()
            except Exception:
                pass


def _refresh_ticket_in_background(key, short_code, queue_number, db_pool, lock) -> None:
    """Re-read one cached ticket; the caller already holds its key lock."""
    try:
        row = _fetch_ticket_row(short_code, queue_number, db_pool)
        now = time.monotonic()
        payload = validate_jwt_token(row["jwt_token"]) if row else None
        with _ticket_cache_lock:
            if row and payload and row.get("status") == "waiting":
                row["jwt_payload"] = payload
                _ticket_cache[key] = (now + _TICKET_CACHE_TTL_SECONDS, dict(row))
            else:
                # No longer a waiting ticket: drop it so the next poll reads
                # the database and reports the real status.
                _ticket_cache.pop(key, None)
    except Exception as e:
        print(f"[JWT] background refresh failed: {e}")
    finally:
        lock.release()


def get_ticket_record_by_short_code(
    short_code: str,
    queue_number: int,
    db_pool,
) -> dict | None:
    key = (str(short_code).strip().upper(), int(queue_number))
    now = time.monotonic()
    cached = _cached_ticket(key, now)
    if cached is not _MISSING:
        return cached

    # Expired but recent: answer from the cache at once and refresh it in the
    # background, so no poll waits on the database at the moment of expiry.
    # A status change (served, no-show, reset) removes the entry outright, so
    # this never answers with a status that staff have already changed.
    stale = _cached_ticket(key, now, allow_stale=True)
    if stale is not _MISSING:
        lock = _ticket_key_lock(key)
        if lock.acquire(blocking=False):
            threading.Thread(
                target=_refresh_ticket_in_background,
                args=(key, short_code, queue_number, db_pool, lock),
                daemon=True,
                name="TicketRefresh",
            ).start()
        return stale

    # One poll per ticket refreshes it; the others arriving at the same moment
    # wait for that answer. Without this, every poller of a ticket missed the
    # cache together the instant it expired and each took a connection. The
    # wait is bounded, so a slow database can never tie up every worker thread.
    lock = _ticket_key_lock(key)
    if not lock.acquire(timeout=_TICKET_LOCK_WAIT_SECONDS):
        stale = _cached_ticket(key, time.monotonic(), allow_stale=True)
        if stale is not _MISSING:
            return stale
        raise TicketLookupBusy()
    try:
        now = time.monotonic()
        cached = _cached_ticket(key, now)
        if cached is not _MISSING:
            return cached
        try:
            row = _fetch_ticket_row(short_code, queue_number, db_pool)
        except Exception as e:
            print(f"[JWT] DB error during validation: {e}")
            # The database is busy: answer with what was known moments ago.
            stale = _cached_ticket(key, now, allow_stale=True)
            if stale is not _MISSING:
                return stale
            raise TicketLookupBusy() from e

        if not row:
            print("[JWT] Short code not found or queue number mismatch")
            with _ticket_cache_lock:
                _ticket_cache[key] = (now + _TICKET_MISS_TTL_SECONDS, None)
            return None
        payload = validate_jwt_token(row["jwt_token"])
        if not payload:
            return None
        row["jwt_payload"] = payload
        if row.get("status") == "waiting":
            with _ticket_cache_lock:
                _ticket_cache[key] = (now + _TICKET_CACHE_TTL_SECONDS, dict(row))
        return row
    finally:
        lock.release()

# TICKET DELETION
def delete_ticket(pdf_path: str) -> bool:
    if not pdf_path:
        return True
    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"[TicketPrinter] 🗑️  Deleted: {os.path.basename(pdf_path)}")
        else:
            print(f"[TicketPrinter] ℹ️  Already removed: {os.path.basename(pdf_path)}")
        return True
    except OSError as e:
        print(f"[TicketPrinter] ❌ Delete failed: {e}")
        return False


def delete_ticket_by_queue_number(queue_number: int, db_pool) -> bool:
    if db_pool is None:
        print(f"[TicketPrinter] ⚠️  No DB pool for Q{queue_number:03d}")
        return False
    try:
        conn   = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT pdf_path FROM queue_records "
            "WHERE queue_number = %s "
            "AND status = 'waiting' "
            "ORDER BY created_at DESC, id DESC "
            "LIMIT 1",
            (queue_number,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if not row or not row.get("pdf_path"):
            return True
        return delete_ticket(row["pdf_path"])
    except Exception as e:
        print(f"[TicketPrinter] ❌ DB error: {e}")
        return False


def delete_all_tickets() -> int:
    deleted = 0
    if not os.path.isdir(TICKETS_OUTPUT_DIR):
        return 0
    for fname in os.listdir(TICKETS_OUTPUT_DIR):
        if fname.lower().endswith(".pdf"):
            if delete_ticket(os.path.join(TICKETS_OUTPUT_DIR, fname)):
                deleted += 1
    print(f"[TicketPrinter] 🗑️  Reset: deleted {deleted} ticket(s)")
    return deleted


# QR CODE GENERATOR (internal)
def _build_qr_image(queue_number: int, short_code: str) -> Image.Image:
    """
    Build a QR code that encodes the ticket status URL.

    URL: {PORTAL_BASE_URL}/api/queue/status?q={queue_number}&token={short_code}

    Scanning this takes the student straight to their live queue status —
    no typing needed. The manual short_code below the QR is the fallback
    for students who cannot scan a code.
    """
    url = (
        f"{PORTAL_BASE_URL}/api/queue/status"
        f"?q={queue_number}&token={short_code}"
    )

    qr = qrcode.QRCode(
        version          = None,
        error_correction = qrcode.constants.ERROR_CORRECT_M,
        box_size         = 5,
        border           = 2,
    )
    qr.add_data(url)
    qr.make(fit=True)

    # Black on white. The old light-purple-on-navy matched the dark ticket but
    # fought the scanner: QR decoders are built for dark modules on a light
    # ground, and low-contrast inverted codes are slower and less reliable to
    # read — especially on a phone camera in a queue area's mixed lighting.
    img = qr.make_image(
        fill_color = "black",
        back_color = "white",
    ).convert("RGB")

    return img


def _pil_to_rl(pil_img: Image.Image) -> ImageReader:
    """Convert PIL Image → ReportLab ImageReader via in-memory PNG bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)



# PDF TICKET GENERATOR
def generate_ticket_pdf(
    queue_number : int,
    short_code   : str,
    position     : int,
    est_wait_min : int,
    service      : str = "Enrollment Office",
    counters_open: int = 2,
    linked_via   : str | None = None,
    student_display_name: str | None = None,
) -> str:
    """
    Render the ticket a student carries away from the queue area.

    `position`, `est_wait_min`, `counters_open` and `linked_via` are still
    accepted — callers in queue_tracker and ticket_service pass them — but are
    deliberately NOT printed. Each one is a snapshot that goes stale the
    moment the paper is cut: the queue keeps moving, so a printed "Position 4"
    or "~12 min" actively misleads the student holding it. Those values are
    served live through the QR link instead. They stay in the signature so
    callers keep working and so the data is available if a future revision
    wants it back.
    """
    # Local import to avoid a circular import — queue_tracker.py imports
    # from this module too (also locally, for the same reason).
    from services.queue_tracker import queue_label
    label = queue_label(queue_number)

    os.makedirs(TICKETS_OUTPUT_DIR, exist_ok=True)

    now       = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename  = f"ticket_{label}_{timestamp}.pdf"
    filepath  = os.path.join(TICKETS_OUTPUT_DIR, filename)

    W = TICKET_WIDTH_MM  * mmUnit
    H = TICKET_HEIGHT_MM * mmUnit
    MARGIN = 6 * mmUnit

    c = pdf_canvas.Canvas(filepath, pagesize=(W, H))

    # Layout walks a cursor DOWN from the top edge rather than hard-coding an
    # absolute offset per element. The student name is optional (walk-in
    # tickets have no student on file), and with fixed offsets every element
    # below it needed a second set of magic numbers for the two cases.
    y = H - 10 * mmUnit

    def rule(gap_before=3.0, gap_after=4.0):
        nonlocal y
        y -= gap_before * mmUnit
        c.setStrokeColor(COLOR_RULE)
        c.setLineWidth(0.5)
        c.line(MARGIN, y, W - MARGIN, y)
        y -= gap_after * mmUnit

    def centred(text, font, size, color, gap_after, shrink_to_fit=False):
        nonlocal y
        if shrink_to_fit:
            # drawCentredString neither wraps nor scales, so a name wider than
            # the ticket silently bleeds off both edges. Measured: a compound
            # surname like "VILLANUEVA-RICAFRENTE, Jose Antonio M." is 68.1mm
            # against 68.0mm of usable width — long names are normal here, not
            # an edge case. Step the size down until it fits, with a floor so
            # it never shrinks into something unreadable.
            usable = W - 2 * MARGIN
            while size > 5.5 and stringWidth(text, font, size) > usable:
                size -= 0.25
        c.setFillColor(color)
        c.setFont(font, size)
        c.drawCentredString(W / 2, y, text)
        y -= gap_after * mmUnit

    # 1. Institution
    centred(INSTITUTION.upper(), "Helvetica", 6, COLOR_MUTED, 4.0)
    centred(service.upper(), "Helvetica", 5.5, COLOR_MUTED, 0)
    rule()

    # 2. Recognized student — omitted entirely for walk-ins, which have no
    #    student on file.
    if student_display_name:
        centred(student_display_name, "Helvetica-Bold", 9, COLOR_INK, 7.0,
                shrink_to_fit=True)

    # 3. Queue number — the one thing the ticket exists to communicate.
    centred("QUEUE NUMBER", "Helvetica", 6.5, COLOR_MUTED, 16.0)
    centred(label, "Helvetica-Bold", 46, COLOR_INK, 0)
    rule(gap_before=6.0)

    # 4. QR code — the live status link. Everything that changes while the
    #    student waits lives behind this, not on the paper.
    centred("SCAN TO CHECK YOUR QUEUE STATUS", "Helvetica", 6, COLOR_MUTED, 3.0)
    qr_size = 34 * mmUnit
    try:
        qr_rl = _pil_to_rl(_build_qr_image(queue_number, short_code))
        c.drawImage(qr_rl, (W - qr_size) / 2, y - qr_size,
                    width=qr_size, height=qr_size)
        y -= qr_size + 5 * mmUnit
    except Exception as e:
        print(f"[TicketPrinter] ⚠️  QR failed: {e}")
        y -= 6 * mmUnit
        centred("[QR unavailable — use the code below]",
                "Helvetica", 6, COLOR_MUTED, 6.0)

    # 5. Short code — the fallback for a student whose phone will not scan.
    centred("OR ENTER THIS CODE", "Helvetica", 5.5, COLOR_MUTED, 7.0)
    centred(short_code, "Helvetica-Bold", 17, COLOR_INK, 0)

    # 6. Footer — issue time only. It is the one detail that stays true after
    #    printing, and it is what staff ask for when resolving a dispute.
    c.setFillColor(COLOR_MUTED)
    c.setFont("Helvetica", 5.5)
    c.drawCentredString(W / 2, 6 * mmUnit,
                        now.strftime("%b %d, %Y  ·  %I:%M %p"))

    c.save()
    print(f"[TicketPrinter] ✅ Saved → {filepath}")
    return filepath

# MAIN ENTRY POINT
def issue_ticket(
    queue_number : int,
    position     : int,
    est_wait_min : int,
    service      : str = "Enrollment Office",
    counters_open: int = 2,
    linked_via   : str | None = None,
    student_display_name: str | None = None,
) -> dict | None:

    try:
        jwt_token  = generate_jwt_token(queue_number, service)
        short_code = generate_short_code()
        expires_at = datetime.now() + timedelta(hours=JWT_EXPIRY_HOURS)
        pdf_path   = generate_ticket_pdf(
            queue_number  = queue_number,
            short_code    = short_code,
            position      = position,
            est_wait_min  = est_wait_min,
            service       = service,
            counters_open = counters_open,
            linked_via    = linked_via,
            student_display_name = student_display_name,
        )
        # Upload the PDF so it outlives the process that made it.
        #
        # The backend runs on Render, whose filesystem is ephemeral: a ticket
        # written to app/tickets/ lives inside the container and disappears on
        # the next restart, redeploy or sleep, and no route serves it, so
        # nobody can fetch it in the meantime. Object storage gives each ticket
        # a URL that survives all of that.
        #
        # database_handler already prefers this over the local path:
        #   ticket.get("storage_url") or ticket["pdf_path"]
        # so queue_records ends up holding the URL whenever the upload worked.
        #
        # Failure is not fatal — upload_ticket_pdf returns None if storage is
        # disabled or unreachable, and the row simply keeps the local path.
        storage = object_storage_service.upload_ticket_pdf(
            pdf_path,
            queue_number,
        ) or {}
        return {
            "queue_number" : queue_number,
            "short_code"   : short_code,
            "jwt_token"    : jwt_token,
            "expires_at"   : expires_at,
            "pdf_path"     : pdf_path,
            # Empty when the upload was skipped or failed, which is exactly
            # what database_handler's `storage_url or pdf_path` expects.
            "storage_key"  : storage.get("storage_key"),
            "storage_url"  : storage.get("storage_url"),
        }
    except Exception as e:
        print(f"[TicketPrinter] ❌ Failed: {e}")
        return None



# STANDALONE TEST
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 55)
    print("QueuEx — Ticket Test (QR + manual code)")
    print("=" * 55)

    ticket = issue_ticket(
        queue_number  = 4,
        position      = 4,
        est_wait_min  = 12,
        service       = "Enrollment Office",
        counters_open = 3,
    )

    if ticket:
        print(f"\n✅ Q{ticket['queue_number']:03d} issued")
        print(f"   Short code : {ticket['short_code']}")
        print(f"   PDF        : {ticket['pdf_path']}")
        print(f"   QR URL     : {PORTAL_BASE_URL}/api/queue/status"
              f"?q={ticket['queue_number']}&token={ticket['short_code']}")

        payload = validate_jwt_token(ticket["jwt_token"])
        if payload:
            print(f"✅ JWT valid — sub={payload['sub']}, jti={payload['jti']}")

        import sys, subprocess
        if sys.platform == "win32":
            os.startfile(ticket["pdf_path"])
        elif sys.platform == "darwin":
            subprocess.run(["open", ticket["pdf_path"]])
        else:
            print(f"   Open manually: {ticket['pdf_path']}")
