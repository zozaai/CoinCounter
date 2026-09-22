import math


def summarize(records, expected):
    scored = [r for r in records if r["actual"] is not None]
    n = len(scored)
    seconds = sum(r["seconds"] for r in records)
    return {
        "scored": n, "completed": len(records), "expected": expected,
        "accuracy": sum(r["count"] == r["actual"] for r in scored) / n if n else None,
        "mae": sum(abs(r["count"] - r["actual"]) for r in scored) / n if n else None,
        "mean_ms": seconds * 1000 / len(records) if records else None,
        "total_seconds": seconds,
        "cached": sum(r["cached"] for r in records),
    }


def pareto_front(rows):
    """Minimize time and maximize accuracy across fully scored, successful runs.

    Equal points remain on the frontier; dominance requires a strict improvement
    in at least one metric without worsening the other.
    """
    eligible = [r for r in rows
                if r["completed"] == r["scored"] == r["expected"] and not r["errors"]
                and all(r[k] is not None and math.isfinite(r[k])
                        for k in ("accuracy", "mean_ms"))]
    frontier = [r for r in eligible if not any(
        other["mean_ms"] <= r["mean_ms"] and other["accuracy"] >= r["accuracy"]
        and (other["mean_ms"] < r["mean_ms"] or other["accuracy"] > r["accuracy"])
        for other in eligible
    )]
    return sorted(frontier, key=lambda r: (r["mean_ms"], -r["accuracy"], r["engine"]))
