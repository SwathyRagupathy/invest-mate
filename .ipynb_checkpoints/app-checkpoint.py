# app.py

import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objects as go
import datetime

# Load data paths
actual_data_path = "data/^GSPC_stock_data_processed.csv"
forecast_path = "data/prophet_forecasts/next_day_forecasts.csv"

app = dash.Dash(__name__)
server = app.server

# Load most recent actual values
def load_actual_data():
    df = pd.read_csv(actual_data_path, parse_dates=["Date"])
    latest = df.iloc[-1]
    return {
        "Open": latest["Open_target"],
        "Close": latest["Close_target"],
        "Gap": latest["Gap_target"],
        "Date": latest["Date"]
    }

# Load forecast data
def load_forecast_data():
    df = pd.read_csv(forecast_path)
    return df

# Plot actual vs predicted
def make_chart(target, actual_series, predicted_point, date_pred):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual_series["Date"],
        y=actual_series[target],
        mode="lines+markers",
        name="Actual",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=[date_pred],
        y=[predicted_point],
        mode="markers+text",
        name="Predicted",
        marker=dict(size=10, color="orange"),
        text=["Predicted"],
        textposition="top center"
    ))

    fig.update_layout(
        title=f"{target} – Actual vs Next-Day Forecast",
        xaxis_title="Date",
        yaxis_title=target,
        height=350
    )
    return fig

# Dashboard layout
app.layout = html.Div([
    html.H1("📈 Prophet Forecast Dashboard – S&P 500"),
    dcc.Interval(id='interval', interval=30*60*1000, n_intervals=0),
    html.Div(id='last-updated'),
    html.Div(id='forecast-table'),
    html.Div(id='forecast-charts')
])

@app.callback(
    [dash.Output('last-updated', 'children'),
     dash.Output('forecast-table', 'children'),
     dash.Output('forecast-charts', 'children')],
    [dash.Input('interval', 'n_intervals')]
)
def update_dashboard(n):
    actuals = load_actual_data()
    forecast_df = load_forecast_data()
    df_actual_full = pd.read_csv(actual_data_path, parse_dates=["Date"]).tail(14)  # Last 14 days

    # Build summary table
    table_rows = []
    for _, row in forecast_df.iterrows():
        target_name = row["target"].replace("_target", "")
        pred_value = row["predicted_value"]
        actual_value = actuals[target_name]
        table_rows.append(html.Tr([
            html.Td(target_name),
            html.Td(f"{pred_value:.2f}"),
            html.Td(f"{actual_value:.2f}"),
            html.Td(str(row["predicted_date"]))
        ]))

    table = html.Table([
        html.Thead(html.Tr([html.Th("Target"), html.Th("Predicted"), html.Th("Actual"), html.Th("Date")])),
        html.Tbody(table_rows)
    ])

    # Line charts
    charts = []
    for _, row in forecast_df.iterrows():
        target_col = row["target"]
        chart = dcc.Graph(
            figure=make_chart(
                target=target_col,
                actual_series=df_actual_full[["Date", target_col]].rename(columns={target_col: target_col}),
                predicted_point=row["predicted_value"],
                date_pred=row["predicted_date"]
            )
        )
        charts.append(chart)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return html.H4(f"Last Updated: {timestamp}"), table, charts

if __name__ == '__main__':
    app.run(debug=True)
