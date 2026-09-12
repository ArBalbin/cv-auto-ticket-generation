import threading
from queue import Queue as ThreadQueue

from core.config import QUEUE_CONFIG
from database.database_handler import get_student_by_id, save_ticket_record
from services.ticket_printer import delete_ticket, issue_ticket


ticket_queue: ThreadQueue = ThreadQueue(maxsize=50)
_worker: threading.Thread | None = None


def enqueue_ticket(
    queue_number: int,
    position: int,
    est_wait_min: float = 0,
    student_id: int | None = None,
    linked_via: str = "manual",
    is_walkin: bool = True,
) -> None:
    ticket_queue.put_nowait({
        "queue_number": queue_number,
        "position": position,
        "est_wait_min": est_wait_min,
        "student_id": student_id,
        "linked_via": linked_via,
        "is_walkin": is_walkin,
    })


def _ticket_worker() -> None:
    """Generate tickets without blocking HTTP requests."""
    while True:
        try:
            job = ticket_queue.get(timeout=1)
        except Exception:
            continue

        qn = job.get("queue_number", "?")
        try:
            from services import queue_service

            if not queue_service.is_queue_number_active(qn):
                print(f"[TicketWorker] Q{qn:03d} skipped - no longer active")
                ticket_queue.task_done()
                continue

            student_display_name = None
            student_id = job.get("student_id")
            if student_id is not None:
                student = get_student_by_id(student_id)
                if student:
                    school_id = student.get("school_id")
                    full_name = student.get("full_name")
                    student_display_name = (
                        f"{school_id} · {full_name}" if full_name else school_id
                    )

            ticket = issue_ticket(
                queue_number=qn,
                position=job["position"],
                est_wait_min=job["est_wait_min"],
                service="Enrollment Office",
                counters_open=QUEUE_CONFIG["num_counters"],
                linked_via=job.get("linked_via"),
                student_display_name=student_display_name,
            )
        except Exception as exc:
            print(f"[TicketWorker] issue_ticket error Q{qn:03d}: {exc}")
            ticket_queue.task_done()
            continue

        if not ticket:
            print(f"[TicketWorker] Ticket generation failed for Q{qn:03d}")
            ticket_queue.task_done()
            continue

        # Everything below must stay inside a guard. An escaping exception
        # here would kill this thread outright, and the failure would be
        # silent: tickets simply stop being generated while the rest of the
        # system keeps reporting healthy. One bad ticket must never take the
        # whole worker down with it.
        try:
            ticket["student_id"] = job.get("student_id")
            ticket["linked_via"] = job.get("linked_via", "manual")
            ticket["is_walkin"] = job.get("is_walkin", True)

            short_code_set = queue_service.queue_tracker.set_short_code(
                qn,
                ticket["short_code"],
            )
            pdf_path_set = queue_service.queue_tracker.set_pdf_path(qn, ticket["pdf_path"])
            if not (short_code_set and pdf_path_set):
                delete_ticket(ticket["pdf_path"])
                print(f"[TicketWorker] Q{qn:03d} discarded - tracker no longer active")
                continue

            print(
                f"[TicketWorker] Q{qn:03d} issued | "
                f"code={ticket['short_code']} | pdf={ticket['pdf_path']}"
            )

            try:
                save_ticket_record(ticket)
            except Exception as db_exc:
                print(f"[TicketWorker] DB save error Q{qn:03d}: {db_exc}")
        except Exception as exc:
            print(f"[TicketWorker] unexpected error finalizing Q{qn:03d}: {exc}")
        finally:
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
