#!/usr/bin/env python3
"""Generate monthly evaluation + equity/capital curve for jma_jma strategy."""

import json
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

JSON_PATH = "report_html/trading_summary_ph1_jma_jma.json"
OUTPUT_DIR = "report_html"
START_EQUITY = 16_500.0


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    trades = data["trades"]
    print(f"Loaded {len(trades)} trades from {JSON_PATH}")

    df = pd.DataFrame(trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df.sort_values("exit_time").reset_index(drop=True)

    # --- Equity / Capital Curve ---
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["equity"] = START_EQUITY + df["cumulative_pnl"]
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["peak"] - df["equity"]
    df["drawdown_pct"] = (df["drawdown"] / df["peak"]) * 100

    max_dd = df["drawdown"].max()
    max_dd_pct = df["drawdown_pct"].max()
    max_dd_idx = df["drawdown"].idxmax()
    max_dd_date = df.loc[max_dd_idx, "exit_time"]

    final_equity = df["equity"].iloc[-1]
    total_return = ((final_equity - START_EQUITY) / START_EQUITY) * 100

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Kapitalkurve (Equity)", "Drawdown (USDT)", "Kumulativer PnL pro Symbol"),
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=df["exit_time"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
            y=df["equity"].values.tolist(),
            mode="lines",
            name="Equity",
            line=dict(color="blue", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,100,255,0.1)",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["exit_time"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
            y=df["peak"].values.tolist(),
            mode="lines",
            name="Peak Equity",
            line=dict(color="green", width=1, dash="dot"),
        ),
        row=1, col=1,
    )
    fig.add_hline(y=START_EQUITY, line_dash="dash", line_color="gray",
                  annotation_text=f"Start: {START_EQUITY:,.0f}", row=1, col=1)

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=df["exit_time"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
            y=(-df["drawdown"]).values.tolist(),
            mode="lines",
            name="Drawdown",
            line=dict(color="red", width=1),
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.3)",
        ),
        row=2, col=1,
    )
    fig.add_annotation(
        x=max_dd_date.strftime("%Y-%m-%d %H:%M"),
        y=-max_dd,
        text=f"Max DD: {max_dd:,.0f} ({max_dd_pct:.1f}%)",
        showarrow=True, arrowhead=2, arrowcolor="red", font=dict(color="red"),
        row=2, col=1,
    )

    # Per-symbol cumulative PnL
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78",
    ]
    for i, symbol in enumerate(sorted(df["symbol"].unique())):
        sym_df = df[df["symbol"] == symbol].copy()
        sym_df["sym_cum_pnl"] = sym_df["pnl"].cumsum()
        fig.add_trace(
            go.Scatter(
                x=sym_df["exit_time"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
                y=sym_df["sym_cum_pnl"].values.tolist(),
                mode="lines",
                name=symbol.replace("/USDC", "").replace("/USDT", ""),
                line=dict(color=colors[i % len(colors)], width=1.5),
            ),
            row=3, col=1,
        )

    fig.update_layout(
        title=dict(
            text=(
                f"JMA→JMA Kapitalkurve | Start: {START_EQUITY:,.0f} → End: {final_equity:,.0f} USDT "
                f"| Return: {total_return:.1f}% | Trades: {len(df)} "
                f"| Max DD: {max_dd:,.0f} ({max_dd_pct:.1f}%)"
            ),
            font=dict(size=14),
        ),
        height=950,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Datum", row=3, col=1)
    fig.update_yaxes(title_text="Equity (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (USDT)", row=2, col=1)
    fig.update_yaxes(title_text="Kum. PnL (USDT)", row=3, col=1)

    eq_path = os.path.join(OUTPUT_DIR, "jmajma_equity_curve.html")
    fig.write_html(eq_path, include_plotlyjs="cdn")
    print(f"Saved equity curve → {eq_path}")

    # --- Monthly Returns ---
    df["month"] = df["exit_time"].dt.to_period("M")
    monthly = df.groupby("month").agg(
        pnl=("pnl", "sum"),
        trades=("pnl", "count"),
        winners=("pnl", lambda x: (x > 0).sum()),
        losers=("pnl", lambda x: (x < 0).sum()),
    ).reset_index()
    monthly["win_rate"] = (monthly["winners"] / monthly["trades"] * 100).round(1)
    monthly["month_str"] = monthly["month"].astype(str)
    monthly["cum_pnl"] = monthly["pnl"].cumsum()
    monthly["equity_eom"] = START_EQUITY + monthly["cum_pnl"]

    bar_colors = ["green" if x >= 0 else "red" for x in monthly["pnl"]]

    total_pnl = monthly["pnl"].sum()
    avg_monthly = monthly["pnl"].mean()
    best = monthly.loc[monthly["pnl"].idxmax()]
    worst = monthly.loc[monthly["pnl"].idxmin()]
    positive_months = (monthly["pnl"] > 0).sum()
    negative_months = (monthly["pnl"] <= 0).sum()

    fig2 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Monatliche PnL (USDT)", "Equity Entwicklung (Monatsende)"),
    )

    fig2.add_trace(
        go.Bar(
            x=monthly["month_str"].tolist(),
            y=monthly["pnl"].values.tolist(),
            marker_color=bar_colors,
            text=[
                f"{p:,.0f}<br>{t} Trades<br>{wr:.0f}% WR"
                for p, t, wr in zip(monthly["pnl"], monthly["trades"], monthly["win_rate"])
            ],
            textposition="outside",
            name="Monatl. PnL",
        ),
        row=1, col=1,
    )

    fig2.add_trace(
        go.Scatter(
            x=monthly["month_str"].tolist(),
            y=monthly["equity_eom"].values.tolist(),
            mode="lines+markers",
            name="Equity (EoM)",
            line=dict(color="blue", width=2),
            marker=dict(size=6),
        ),
        row=2, col=1,
    )
    fig2.add_hline(y=START_EQUITY, line_dash="dash", line_color="gray",
                   annotation_text=f"Start: {START_EQUITY:,.0f}", row=2, col=1)

    fig2.update_layout(
        title=dict(
            text=(
                f"JMA→JMA Monatliche Auswertung | Total: {total_pnl:,.0f} USDT "
                f"| Avg: {avg_monthly:,.0f}/Monat | Best: {best['month_str']} ({best['pnl']:,.0f}) "
                f"| Worst: {worst['month_str']} ({worst['pnl']:,.0f}) "
                f"| Pos: {positive_months} / Neg: {negative_months} Monate"
            ),
            font=dict(size=13),
        ),
        height=700,
        showlegend=True,
        template="plotly_white",
    )
    fig2.update_xaxes(title_text="Monat", row=2, col=1)
    fig2.update_yaxes(title_text="PnL (USDT)", row=1, col=1)
    fig2.update_yaxes(title_text="Equity (USDT)", row=2, col=1)

    monthly_path = os.path.join(OUTPUT_DIR, "jmajma_monthly_returns.html")
    fig2.write_html(monthly_path, include_plotlyjs="cdn")
    print(f"Saved monthly returns → {monthly_path}")

    # --- Console Summary ---
    print("\n" + "=" * 80)
    print(f"JMA→JMA AUSWERTUNG  ({data['start']} bis {data['end']})")
    print("=" * 80)
    print(f"Start-Kapital:    {START_EQUITY:>12,.0f} USDT")
    print(f"End-Kapital:      {final_equity:>12,.0f} USDT")
    print(f"Total PnL:        {total_pnl:>12,.0f} USDT")
    print(f"Return:           {total_return:>11.1f}%")
    print(f"Max Drawdown:     {max_dd:>12,.0f} USDT ({max_dd_pct:.1f}%)")
    print(f"Trades:           {len(df):>12,}")
    print(f"Win-Rate:         {data['win_rate_pct']:>11.1f}%")
    print(f"Avg Trade PnL:    {data['avg_trade_pnl']:>12,.2f} USDT")
    print(f"Pos. Monate:      {positive_months:>12} / {len(monthly)}")
    print(f"Neg. Monate:      {negative_months:>12} / {len(monthly)}")
    print()

    print("MONATLICHE AUFSTELLUNG:")
    print(f"{'Monat':<10} {'PnL':>12} {'Trades':>8} {'WR%':>6} {'Kum.PnL':>14} {'Equity':>14}")
    print("-" * 66)
    for _, row in monthly.iterrows():
        print(
            f"{row['month_str']:<10} {row['pnl']:>12,.0f} {row['trades']:>8} "
            f"{row['win_rate']:>5.1f}% {row['cum_pnl']:>14,.0f} {row['equity_eom']:>14,.0f}"
        )
    print("-" * 66)
    print(f"{'TOTAL':<10} {total_pnl:>12,.0f} {monthly['trades'].sum():>8}")


if __name__ == "__main__":
    main()
