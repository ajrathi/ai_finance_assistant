"""Plotly chart wrappers for the Finnie UI."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def portfolio_pie_chart(holdings: List[Dict]) -> go.Figure:
    """Render a portfolio allocation pie chart."""
    if not holdings:
        fig = go.Figure()
        fig.add_annotation(text="No holdings to display", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    labels = [h.get("ticker", "?") for h in holdings]
    values = [h.get("value", 0) for h in holdings]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.35,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Share: %{percent}<extra></extra>",
    )])
    fig.update_layout(
        title="Portfolio Allocation",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=50, b=50, l=20, r=20),
    )
    return fig


def portfolio_bar_chart(holdings: List[Dict]) -> go.Figure:
    """Render a horizontal bar chart of portfolio holdings by value."""
    if not holdings:
        return go.Figure()

    df = pd.DataFrame(holdings)
    if "value" not in df.columns:
        return go.Figure()

    df = df.sort_values("value", ascending=True)
    fig = px.bar(
        df,
        x="value",
        y="ticker",
        orientation="h",
        title="Holdings by Value",
        labels={"value": "Value ($)", "ticker": "Ticker"},
        color="value",
        color_continuous_scale="Blues",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(margin=dict(t=50, b=30, l=80, r=20))
    return fig


def goal_projection_chart(
    current_savings: float,
    monthly_contribution: float,
    time_horizon_years: int,
    target_amount: float,
    rates: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """Render compound interest projection chart with multiple scenarios."""
    if rates is None:
        rates = {"Conservative (5%)": 0.05, "Moderate (7%)": 0.07, "Optimistic (9%)": 0.09}

    months = time_horizon_years * 12
    t = list(range(months + 1))
    years_axis = [m / 12 for m in t]

    fig = go.Figure()

    colors = {"Conservative (5%)": "#5B8DB8", "Moderate (7%)": "#2E7D32", "Optimistic (9%)": "#E65100"}

    for label, rate in rates.items():
        monthly_rate = rate / 12
        values = []
        for m in t:
            fv_principal = current_savings * ((1 + monthly_rate) ** m)
            if monthly_rate > 0:
                fv_contrib = monthly_contribution * (((1 + monthly_rate) ** m - 1) / monthly_rate)
            else:
                fv_contrib = monthly_contribution * m
            values.append(fv_principal + fv_contrib)

        fig.add_trace(go.Scatter(
            x=years_axis,
            y=values,
            name=label,
            mode="lines",
            line=dict(width=2, color=colors.get(label, None)),
            hovertemplate=f"{label}<br>Year %{{x:.1f}}: $%{{y:,.0f}}<extra></extra>",
        ))

    # Target line
    fig.add_hline(
        y=target_amount,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Target: ${target_amount:,.0f}",
        annotation_position="right",
    )

    fig.update_layout(
        title=f"Goal Projection over {time_horizon_years} Years",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(t=60, b=80, l=80, r=80),
        hovermode="x unified",
    )

    # Annotation: educational disclaimer
    fig.add_annotation(
        text="For illustrative purposes only. Not a guarantee of returns.",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="center",
    )

    return fig


def market_index_bar(market_data: Dict[str, Dict]) -> go.Figure:
    """Render a bar chart of major index daily changes."""
    if not market_data:
        fig = go.Figure()
        fig.add_annotation(text="Market data unavailable", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    names = list(market_data.keys())
    changes = [market_data[n].get("change_pct", 0) for n in names]
    colors = ["#2E7D32" if c >= 0 else "#C62828" for c in changes]

    fig = go.Figure(data=[go.Bar(
        x=names,
        y=changes,
        marker_color=colors,
        text=[f"{c:+.2f}%" for c in changes],
        textposition="outside",
        hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
    )])
    fig.update_layout(
        title="Major Index Performance (Today)",
        yaxis_title="% Change",
        yaxis_tickformat="+.2f%",
        margin=dict(t=60, b=40, l=60, r=20),
    )
    fig.add_hline(y=0, line_color="gray", line_width=0.5)
    return fig
