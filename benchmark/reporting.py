"""ASCII tables, machine-readable summaries, and interactive Plotly comparisons."""
import csv
from html import escape
import io
import math
from pathlib import Path

from .metrics import pareto_front
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


def accuracy_time_figure(summary):
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    rows = summary["engines"]
    plotted = [r for r in rows if all(r[k] is not None and math.isfinite(r[k])
                                    for k in ("accuracy", "mean_ms"))]
    frontier = pareto_front(rows)
    figure = go.Figure()
    colors = qualitative.Dark24
    x_max = max((r["mean_ms"] for r in plotted), default=1)
    x_max = x_max * 1.12 if x_max else 1

    if frontier:
        # Models with identical metrics share a vertex, but retain their circles.
        points = list(dict.fromkeys((r["mean_ms"], r["accuracy"]) for r in frontier))
        figure.add_trace(go.Scatter(
            x=[p[0] for p in points], y=[p[1] for p in points], mode="lines",
            name="Pareto frontier", line={"color": "#475569", "width": 2},
            hoverinfo="skip",
        ))
    for index, row in enumerate(plotted):
        color = colors[index % len(colors)]
        complete = row["completed"] == row["scored"] == row["expected"] and not row["errors"]
        label = row["engine"] + ("" if complete else " (partial)")
        figure.add_trace(go.Scatter(
            x=[row["mean_ms"]], y=[row["accuracy"]], name=label,
            mode="markers+text", text=[label], textposition="top center",
            cliponaxis=False,
            marker={"symbol": "circle", "size": 16, "color": color,
                    "opacity": 1 if complete else 0.45,
                    "line": {"color": "white", "width": 1}},
            customdata=[[row["scored"], row["expected"], row["mae"], row["total_seconds"],
                         row["cached"], len(row["errors"])]],
            hovertemplate=("<b>%{fullData.name}</b><br>Mean ms: %{x:.3f}"
                           "<br>Accuracy: %{y:.2%}<br>Scored: %{customdata[0]}/%{customdata[1]}"
                           "<br>MAE: %{customdata[2]:.3f}<br>Total sec: %{customdata[3]:.6f}"
                           "<br>Cached: %{customdata[4]}<br>Failed: %{customdata[5]}<extra></extra>"),
        ))
    figure.update_layout(
        title=f"Accuracy vs. processing time — {escape(summary['split'])}",
        template="plotly_white",
        xaxis={"title": "Processing time (Mean ms)", "range": [0, x_max],
               "ticklen": 5, "tickcolor": "#64748b"},
        yaxis={"title": "Accuracy", "range": [-0.04, 1.08], "tickformat": ".0%",
               "ticklen": 5, "tickcolor": "#64748b"},
        legend={"title": "Models"},
        margin={"l": 90, "r": 60, "t": 80, "b": 90},
    )
    note = "Pareto frontier: lower time, higher accuracy; fully scored, error-free runs only."
    omitted = [r["engine"] for r in rows if r not in plotted]
    if omitted:
        note += "<br>No accuracy/time data: " + escape(", ".join(omitted))
    figure.add_annotation(text=note, x=0, y=-0.17, xref="paper", yref="paper",
                          xanchor="left", showarrow=False, align="left")
    if not plotted:
        figure.add_annotation(text="No scored models with timing data.", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False)
    return figure


def save_comparison(folder, summary, stem=None):
    """Save the table metrics and a self-contained chart that works offline."""
    folder = Path(folder)
    summary_name = f"{stem}_summary.json" if stem else "summary.json"
    chart_name = f"{stem}_accuracy_vs_time.html" if stem else "accuracy_vs_time.html"
    summary_path = folder / summary_name
    chart_path = folder / chart_name
    write_json(summary_path, summary)
    figure = accuracy_time_figure(summary)
    atomic_write(chart_path, figure.to_html(
        full_html=True, include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    ))
    return summary_path, chart_path
