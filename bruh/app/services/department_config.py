"""Canonical department values shared by profile validation and predictions."""

DEPARTMENTS = ("CCS", "CBM", "CHS", "CCJE", "COE", "CTED")
DEPARTMENT_SET = frozenset(DEPARTMENTS)


def normalize_department(value: str | None) -> str | None:
    if value is None:
        return None
    code = value.strip().upper()
    if code not in DEPARTMENT_SET:
        raise ValueError("department must be one of: " + ", ".join(DEPARTMENTS))
    return code
