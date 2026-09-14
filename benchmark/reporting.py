"""ASCII reporting. Future Plotly rendering will consume the same summaries."""
import csv
import io

from .results import atomic_write, write_json


def table(rows, sort="accuracy"):
    def key(row):
        incomplete = row["completed"] != row["expected"]
        accuracy = row["accuracy"] if row["accuracy"] is not None else -1
        time = row["mean_ms"] if row["mean_ms"] is not None else float("inf")
        return (incomplete, time, -accuracy) if sort == "time" else (incomplete, -accuracy, time)
    headers = ["Engine", "Scored", "Accuracy", "MAE", "Mean ms", "Total sec", "Cached", "Failed"]
    data = []
    for r in sorted(rows, key=key):
        data.append([r["engine"], f"{r['scored']}/{r['expected']}",
                     f"{r['accuracy']:.2%}" if r["accuracy"] is not None else "N/A",
                     f"{r['mae']:.3f}" if r["mae"] is not None else "N/A",
                     f"{r['mean_ms']:.3f}" if r["mean_ms"] is not None else "N/A",
                     f"{r['total_seconds']:.6f}", str(r["cached"]), str(len(r["errors"]))])
    widths = [max(len(str(row[i])) for row in [headers] + data) for i in range(len(headers))]
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def line(row):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |"
    return "\n".join([border, line(headers), border] + [line(r) for r in data] + [border])


def save_summary(folder, summary):
    write_json(folder / "summary.json", summary)
    output = io.StringIO()
    rows = [{k: v for k, v in row.items() if k != "errors"} for row in summary["engines"]]
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(folder / "summary.csv", output.getvalue())
