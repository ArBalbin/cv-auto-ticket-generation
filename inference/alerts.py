import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time

LAST_ALERT_TIME = 0
COOLDOWN = 60  # seconds

def send_email_alert(count):
    global LAST_ALERT_TIME

    if time.time() - LAST_ALERT_TIME < COOLDOWN:
        print("⏳ Alert skipped (cooldown)")
        return

    LAST_ALERT_TIME = time.time()

    sender = "safetycrowd@gmail.com"
    password = "ysjz lbkt wukp grfe"  # App Password (keep secret ❗)

    receivers = [
        "mayankchandel830@gmail.com",
        "ayushgr2811@gmail.com",
        "n.hemunegi11@gmail.com",
        "ayushnainwal135@gmail.com"
    ]

    print("📧 Trying to send email...")

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = "🚨 Crowd Risk Alert - Immediate Action Required"

    body = f"""
Dear Authority Team,

This is an automated notification from the Smart Crowd Monitoring System.

⚠ ALERT DETAILS
----------------------------------------
• People Count      : {count}
• Risk Level        : HIGH
• Time Detected     : {timestamp}
• Monitoring Source : Camera 1

The crowd density has exceeded the safe threshold. Immediate preventive action is recommended to avoid potential safety risks.

Please treat this alert as high priority.

Regards,
Smart Crowd Monitoring System
AI-Based Surveillance Unit

(Note: This is an auto-generated email. Please do not reply.)
"""

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(receivers)
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receivers, msg.as_string())
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print("❌ Email error:", e)
