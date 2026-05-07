import os
import io
import jwt
import secrets
import string
import qrcode
from datetime import datetime, timezone, timedelta
from PIL import Image

from reportlab.lib.units import mm as mmUnit
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
TICKET_HEIGHT_MM = 175   # taller than before to fit QR + manual input

INSTITUTION = "Naga College Foundation, Inc."
SYSTEM_NAME = "QueueFlow"

# Base URL for QR code — student scans and lands on their live status page.
# Set PORTAL_BASE_URL in .env for your server IP or domain.
# e.g. http://192.168.1.10:5000  or  https://queueflow.ncf.edu
JWT_ALGORITHM    = "HS256"
JWT_EXPIRY_HOURS = 4

COLOR_DARK   = HexColor("#1A1A2E")
COLOR_PURPLE = HexColor("#7F77DD")
COLOR_LIGHT  = HexColor("#A29EF0")
COLOR_WHITE  = HexColor("#FFFFFF")
COLOR_GRAY   = HexColor("#AAAAAA")
COLOR_BG     = HexColor("#F4F4F8")


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
        conn   = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT jwt_token FROM queue_records "
            "WHERE short_code = %s "
            "AND queue_number = %s "
            "AND status = 'waiting' "
            "AND (expires_at IS NULL OR expires_at > NOW()) "
            "ORDER BY created_at DESC, id DESC "
            "LIMIT 1",
            (short_code, queue_number)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if not row:
            print("[JWT] Short code not found or queue number mismatch")
            return None
        return validate_jwt_token(row["jwt_token"])
    except Exception as e:
        print(f"[JWT] DB error during validation: {e}")
        return None

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

    img = qr.make_image(
        fill_color = "#A29EF0",   # light purple — visible on dark background
        back_color = "#1A1A2E",   # dark navy — matches ticket background
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
) -> str:
   
    os.makedirs(TICKETS_OUTPUT_DIR, exist_ok=True)

    now       = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename  = f"ticket_Q{queue_number:03d}_{timestamp}.pdf"
    filepath  = os.path.join(TICKETS_OUTPUT_DIR, filename)

    W = TICKET_WIDTH_MM  * mmUnit
    H = TICKET_HEIGHT_MM * mmUnit

    c = pdf_canvas.Canvas(filepath, pagesize=(W, H))

    #1. Background
    c.setFillColor(COLOR_DARK)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    #2. Institution header
    c.setFillColor(COLOR_GRAY)
    c.setFont("Helvetica", 6)
    c.drawCentredString(W / 2, H - 12 * mmUnit, INSTITUTION.upper())
    c.setFont("Helvetica", 5.5)
    c.drawCentredString(W / 2, H - 17 * mmUnit, service.upper())

    # Top divider
    c.setStrokeColor(HexColor("#333355"))
    c.setDash(3, 3)
    c.line(5 * mmUnit, H - 20 * mmUnit, W - 5 * mmUnit, H - 20 * mmUnit)
    c.setDash()

    #3. Queue number
    c.setFont("Helvetica", 7)
    c.drawCentredString(W / 2, H - 27 * mmUnit, "QUEUE NUMBER")

    c.setFillColor(COLOR_LIGHT)
    c.setFont("Helvetica-Bold", 52)
    c.drawCentredString(W / 2, H - 48 * mmUnit, f"Q{queue_number:03d}")

    # Position badge
    bw, bh = 38 * mmUnit, 7 * mmUnit
    bx, by = (W - bw) / 2, H - 54 * mmUnit
    c.setFillColor(HexColor("#2A2A4A"))
    c.roundRect(bx, by, bw, bh, 3 * mmUnit, fill=1, stroke=0)
    c.setFillColor(COLOR_LIGHT)
    c.setFont("Helvetica-Bold", 7)
    c.drawCentredString(W / 2, by + 2 * mmUnit, f"Waiting  ·  Position {position}")

    # Tear line 
    c.setDash(3, 3)
    c.line(5 * mmUnit, H - 62 * mmUnit, W - 5 * mmUnit, H - 62 * mmUnit)
    c.setDash()
    c.setFillColor(COLOR_BG)
    c.circle(0, H - 62 * mmUnit, 3 * mmUnit, fill=1, stroke=0)
    c.circle(W, H - 62 * mmUnit, 3 * mmUnit, fill=1, stroke=0)

    #4. QR label 
    c.setFillColor(COLOR_GRAY)
    c.setFont("Helvetica", 6)
    c.drawCentredString(W / 2, H - 66 * mmUnit, "SCAN TO CHECK YOUR QUEUE STATUS")

    # 5. QR code 
    try:
        qr_img  = _build_qr_image(queue_number, short_code)
        qr_rl   = _pil_to_rl(qr_img)
        qr_size = 36 * mmUnit                  # square QR on ticket
        qr_x    = (W - qr_size) / 2            # centred
        qr_y    = H - 104 * mmUnit             # bottom-left corner of QR
        c.drawImage(qr_rl, qr_x, qr_y, width=qr_size, height=qr_size)
    except Exception as e:
        print(f"[TicketPrinter] ⚠️  QR failed: {e}")
        c.setFillColor(COLOR_GRAY)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W / 2, H - 85 * mmUnit, "[QR unavailable — use code below]")

    # 6. Divider + OR ENTER MANUALLY 
    c.setStrokeColor(HexColor("#333355"))
    c.setDash(2, 2)
    c.line(10 * mmUnit, H - 106 * mmUnit, W - 10 * mmUnit, H - 106 * mmUnit)
    c.setDash()

    c.setFillColor(HexColor("#555577"))
    c.setFont("Helvetica", 5.5)
    c.drawCentredString(W / 2, H - 110 * mmUnit, "OR ENTER CODE MANUALLY AT THE PORTAL")

    # 7. Token box with short code 
    c.setFillColor(HexColor("#22224A"))
    c.roundRect(
        5 * mmUnit, H - 124 * mmUnit,
        W - 10 * mmUnit, 12 * mmUnit,
        3 * mmUnit, fill=1, stroke=0
    )

    # "ACCESS TOKEN" micro-label
    c.setFillColor(COLOR_GRAY)
    c.setFont("Helvetica", 5.5)
    c.drawCentredString(W / 2, H - 115 * mmUnit, "ACCESS TOKEN")

    # Short code — large, unambiguous font
    c.setFillColor(COLOR_PURPLE)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(W / 2, H - 122 * mmUnit, short_code)

    # 8. Portal URL hint 
    portal_display = PORTAL_BASE_URL.replace("http://", "").replace("https://", "")
    c.setFillColor(HexColor("#666688"))
    c.setFont("Helvetica", 5)
    c.drawCentredString(W / 2, H - 127 * mmUnit, f"Portal: {portal_display}")

    # 9. JWT secured badge
    c.setFillColor(HexColor("#1A1A3A"))
    c.roundRect(
        W / 2 - 16 * mmUnit, H - 133 * mmUnit,
        32 * mmUnit, 5 * mmUnit,
        2 * mmUnit, fill=1, stroke=0
    )
    c.setFillColor(HexColor("#534AB7"))
    c.setFont("Helvetica-Bold", 5)
    c.drawCentredString(
        W / 2, H - 131 * mmUnit,
        f"JWT SECURED  ·  EXPIRES IN {JWT_EXPIRY_HOURS} HOURS"
    )

    # 10. Detail rows
    # ── FIX: dt was undefined — set starting Y just below the JWT badge ───────
    # JWT badge bottom sits at H-133mm; 6mm gap gives dt = H-139mm.
    # With lg=6mm and 5 rows the last row lands at H-163mm, leaving 7mm
    # of breathing room before the bottom tear line at H-170mm.
    lx = 8  * mmUnit
    rx = W  - 8 * mmUnit
    lg = 6  * mmUnit
    dt = H  - 139 * mmUnit   # ← was missing; caused NameError on every ticket

    def row(label, value, y):
        c.setFillColor(COLOR_GRAY)
        c.setFont("Helvetica", 6.5)
        c.drawString(lx, y, label)
        c.setFillColor(COLOR_WHITE)
        c.setFont("Helvetica-Bold", 6.5)
        c.drawRightString(rx, y, value)
        c.setStrokeColor(HexColor("#2A2A4A"))
        c.setLineWidth(0.3)
        c.line(lx, y - 1.5 * mmUnit, rx, y - 1.5 * mmUnit)

    row("Issued",        now.strftime("%b %d, %Y"),       dt)
    row("Time",          now.strftime("%I:%M %p"),         dt - lg)
    row("Est. wait",     f"~{est_wait_min} min",           dt - lg * 2)
    row("Counters open", str(counters_open),               dt - lg * 3)
    row("Token expires", f"in {JWT_EXPIRY_HOURS} hrs",     dt - lg * 4)

    # 11. Bottom tear line + footer
    c.setStrokeColor(HexColor("#333355"))
    c.setDash(3, 3)
    c.line(5 * mmUnit, H - 170 * mmUnit, W - 5 * mmUnit, H - 170 * mmUnit)
    c.setDash()
    c.setFillColor(COLOR_BG)
    c.circle(0, H - 170 * mmUnit, 3 * mmUnit, fill=1, stroke=0)
    c.circle(W, H - 170 * mmUnit, 3 * mmUnit, fill=1, stroke=0)

    c.setFillColor(HexColor("#555577"))
    c.setFont("Helvetica", 5.5)
    c.drawCentredString(W / 2, H - 174 * mmUnit,
                        "Please keep this ticket. Scan QR or type code.")
    c.setFillColor(HexColor("#333355"))
    c.setFont("Helvetica", 5)
    c.drawCentredString(W / 2, H - 178 * mmUnit,
                        f"{SYSTEM_NAME}  |  NCF  |  {now.year}")

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
        )
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
            "storage_key"   : storage.get("storage_key"),
            "storage_url"   : storage.get("storage_url"),
        }
    except Exception as e:
        print(f"[TicketPrinter] ❌ Failed: {e}")
        return None



# STANDALONE TEST
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 55)
    print("QueueFlow — Ticket Test (QR + manual code)")
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
