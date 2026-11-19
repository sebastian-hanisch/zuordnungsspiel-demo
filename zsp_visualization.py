"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import zsp_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_matrix(w, matching, load_names, truck_names):
    """Paarwerte als Heatmap; die Paare der optimalen Zuordnung sind mit einem Rahmen markiert. Beschriftung als Annotation (kein Text-Trace)."""
    a, b = len(w), len(w[0])
    fig = go.Figure(go.Heatmap(z=[list(map(float, row)) for row in w], x=truck_names, y=load_names, colorscale=[[0.0, "#f4f8fc"], [1.0, "#8fb8dc"]], showscale=False,
                               hovertemplate="%{y} und %{x}: %{z:.2f} €<extra></extra>", xgap=3, ygap=3))
    for i in range(a):
        for j in range(b):
            fig.add_annotation(x=truck_names[j], y=load_names[i], text=f"{w[i][j]:.2f}".replace(".", ","), showarrow=False, font=dict(size=13))
    for i, j in matching:
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line=dict(color=C.COLOR_BAD, width=3))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category", side="top")
    return _base(fig, height=max(240, 70 * a + 90))


def build_core_ranges(names, a, low, high, point=None, shapley=None):
    """Spanne des Kerns je Spieler (von der niedrigeren zur höheren Ecke) als dicke Linie; Raute: gewählter Kernpunkt, Stern: Shapley-Wert."""
    fig = go.Figure()
    for idx, name in enumerate(names):
        color = C.COLOR_LOAD if idx < a else C.COLOR_TRUCK
        fig.add_trace(go.Scatter(x=[low[idx], high[idx]], y=[name, name], mode="lines", line=dict(color=color, width=14), showlegend=False,
                                 hovertemplate=f"{name}: von %{{x:.2f}} €<extra></extra>"))
        if low[idx] == high[idx]:
            fig.add_trace(go.Scatter(x=[low[idx]], y=[name], mode="markers", marker=dict(color=color, size=12), showlegend=False,
                                     hovertemplate=f"{name}: genau %{{x:.2f}} €<extra></extra>"))
    if point is not None:
        fig.add_trace(go.Scatter(x=list(point), y=names, mode="markers", name="gewählter Kernpunkt", marker=dict(symbol="diamond", size=11, color="#222222"),
                                 hovertemplate="%{y}: %{x:.2f} €<extra></extra>"))
    if shapley is not None:
        fig.add_trace(go.Scatter(x=list(shapley), y=names, mode="markers", name="Shapley-Wert", marker=dict(symbol="star", size=13, color=C.COLOR_SHAPLEY),
                                 hovertemplate="%{y}: %{x:.2f} €<extra></extra>"))
    fig.update_yaxes(type="category", autorange="reversed")
    fig.update_xaxes(title_text="Auszahlung in €", rangemode="tozero")
    return _base(fig, height=max(260, 42 * len(names) + 110))


def build_auction(history, truck_names):
    """Preise der Lkw über die Runden (Treppenlinien)."""
    fig = go.Figure()
    rounds = list(range(len(history)))
    for j, name in enumerate(truck_names):
        fig.add_trace(go.Scatter(x=rounds, y=[float(p[j]) for p in history], mode="lines", line=dict(shape="hv", width=3), name=name,
                                 hovertemplate="Runde %{x}: %{y:.0f} €<extra>" + name + "</extra>"))
    fig.update_xaxes(title_text="Runde")
    fig.update_yaxes(title_text="Preis in €", rangemode="tozero")
    return _base(fig)


def build_share(rows, dist_names):
    """Experiment 1: Anteil Shapley im Kern über k, eine Linie je Verteilung."""
    fig = go.Figure()
    for dist in dict.fromkeys(r["dist"] for r in rows):
        sel = [r for r in rows if r["dist"] == dist]
        fig.add_trace(go.Scatter(x=[r["k"] for r in sel], y=[100.0 * r["share"] for r in sel], mode="lines+markers", name=dist_names.get(dist, dist),
                                 hovertemplate="k = %{x}: %{y:.1f} %<extra>" + dist_names.get(dist, dist) + "</extra>"))
    fig.update_xaxes(title_text="Märkte k × k", dtick=1)
    fig.update_yaxes(title_text="Shapley im Kern (%)", range=[-2, 102])
    return _base(fig, height=400)


def build_density(rows):
    """Experiment 3a: Anteil Shapley im Kern gegen die Dichte."""
    fig = go.Figure(go.Bar(x=[f"{r['q']} %" for r in rows], y=[100.0 * r["share"] for r in rows], marker_color=C.COLOR_LINE,
                           hovertemplate="Dichte %{x}: %{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Anteil der Paare mit Wert ungleich 0", type="category")
    fig.update_yaxes(title_text="Shapley im Kern (%)", range=[0, 100])
    return _base(fig)


def build_width(rows, dist_names):
    """Experiment 4: mittlere Kernbreite gegen k."""
    fig = go.Figure()
    for dist in dict.fromkeys(r["dist"] for r in rows):
        sel = [r for r in rows if r["dist"] == dist]
        fig.add_trace(go.Scatter(x=[str(r["k"]) for r in sel], y=[r["width"] for r in sel], mode="lines+markers", name=dist_names.get(dist, dist),
                                 hovertemplate="k = %{x}: %{y:.2f}<extra>" + dist_names.get(dist, dist) + "</extra>"))
    fig.update_xaxes(title_text="Märkte k × k", type="category")
    fig.update_yaxes(title_text="Kernbreite je Spieler / (v(N)/k)", rangemode="tozero")
    return _base(fig)
