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
