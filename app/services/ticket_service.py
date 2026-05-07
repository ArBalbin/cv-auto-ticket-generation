import threading
from queue import Queue as ThreadQueue

from core.config import QUEUE_CONFIG
from database.database_handler import save_ticket_record
from services.ticket_printer import delete_ticket, issue_ticket


ticket_queue: ThreadQueue = ThreadQueue(maxsize=50)
_worker: threading.Thread | None = None


def enqueue_ticket(queue_number: int, position: int, est_wait_min: float = 0) -> None:
    ticket_queue.put_nowait({
        "queue_number": queue_number,
        "position": position,
        "est_wait_min": est_wait_min,
    })


def _ticket_worker() -> None:
    """Generate tickets without blocking HTTP requests."""
    while True:
        try:
            job = ticket_queue.get(timeout=1)
        except Exception:
            continue

        qn = job["queue_number"]
        try:
            from services import queue_service

            if not queue_service.is_queue_number_active(qn):
                print(f"[TicketWorker] Q{qn:03d} skipped - no longer active")
                ticket_queue.task_done()
                continue

            ticket = issue_ticket(
                queue_number=qn,
                position=job["position"],
                est_wait_min=job["est_wait_min"],
                service="Enrollment Office",
                counters_open=QUEUE_CONFIG["num_counters"],
            )
        except Exception as exc:
            print(f"[TicketWorker] issue_ticket error Q{qn:03d}: {exc}")
            ticket_queue.task_done()
            continue

        if not ticket:
            print(f"[TicketWorker] Ticket generation failed for Q{qn:03d}")
            ticket_queue.task_done()
            continue

        short_code_set = queue_service.queue_tracker.set_short_code(
            qn,
            ticket["short_code"],
        )
        pdf_path_set = queue_service.queue_tracker.set_pdf_path(qn, ticket["pdf_path"])
        if not (short_code_set and pdf_path_set):
            delete_ticket(ticket["pdf_path"])
            print(f"[TicketWorker] Q{qn:03d} discarded - tracker no longer active")
            ticket_queue.task_done()
            continue

        print(
            f"[TicketWorker] Q{qn:03d} issued | "
            f"code={ticket['short_code']} | pdf={ticket['pdf_path']}"
        )

        save_ticket_record(ticket)
        ticket_queue.task_done()


def start_ticket_worker() -> None:
    global _worker
    if _worker and _worker.is_alive():
        return
    _worker = threading.Thread(
        target=_ticket_worker,
        daemon=True,
        name="TicketWorker",
    )
    _worker.start()
    print("[TicketWorker] Started")


start_ticket_worker()
