#!/usr/bin/env python3
"""MeetUp 053 - Beginners Python and Machine Learning - 31st Mar 2020 - Charting COVID-19 doubling rate with plotly

Youtube: https://youtu.be/pXctRtdlxCM
Colab:   https://colab.research.google.com/drive/10QrPb0ion0vCBnp1HtxtTcCD0SXaeGaQ
Github:  https://github.com/timcu/bpaml-sessions/tree/master/online
MeetUp:  https://www.meetup.com/beginners-python-machine-learning/events/269665540/

Learning objectives:
- pandas DataFrames and Series
- plotly.py

Requirements:
numpy
pandas
plotly

@author D Tim Cummings
"""

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px


# https://plotly.com/python/creating-and-updating-figures/

# plotly.py is a library for sending JSON objects to plotly.js
# at a low level we can create a dict and send it straight to plotly.js
fig = {
    "data": [{"type": "scatter", "x": [1, 2, 3], "y": [1, 3, 2], "name": "up down"},
             {"type": "scatter", "x": [1, 2, 4], "y": [1, 2.5, 3.5], "name": "climber"}
             ],
    "layout": {"title": {"text": "Scatter chart constructed as a dict"}}
}
# The method in the next line works out we are using colab and uses a colab renderer to
# implement plotly.js in colab and display our interactive chart (try hovering and clicking)
# plotly.io.show(fig)
# If you are not using interactive python you can create an html file and open it
plotly.io.write_html(fig, "demo1.html", auto_open=True)

# Challenge 1: Given the following lists of x and y values which represent sigmoid function
# Plot the values in a scatter chart using dict
x = np.linspace(-10, 10, 21)
y = 1 / (1 + np.exp(-x))
print(x)
print(y)

# Solution to challenge 1
# if you are only showing one item in data list, you need to explicitly show the legend
fig = {
    "data": [{"type": "scatter", "x": x, "y": y, "name": "sigmoid", "showlegend": True}],
    "layout": {"title": {"text": "Sigmoid function constructed as a dict"}}
}
# plotly.io.show(fig)
plotly.io.write_html(fig, "challenge1.html", auto_open=True)

# At a higher level we can use plotly graph_objects which have a built-in validation
fig = go.Figure(
    data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2], name="blue boxes", showlegend=True, )],
    layout=go.Layout(
        title=go.layout.Title(text="Bar chart constructed using graph objects")
    )
)
# Figure has a "write_html()" method for those not using interactive python
fig.write_html("demo2.html", auto_open=True)
# In Google colab we can call "show()" method
# fig.show()

# print fig to see how graph_objects are converted to a dict
print(fig)

# use to_dict() to see the full dictionary
fig.to_dict()

# Challenge 2: Using x and y from Challenge 1 plot the values using graph_objects

# Solution to challenge 2
# Also demonstrates how to use lines or markers or both and how to set marker symbols
fig = go.Figure(
    data=[go.Scatter(x=x, y=y, name="sigmoid", showlegend=True, mode="lines+markers", marker_symbol="hash-dot",
                     marker_line_width=1, marker_size=15)],
    layout=go.Layout(
        title=go.layout.Title(text="Sigmoid function constructed from graph_object")
    )
)
# see the created dict. Notice what happens with marker_line_width
print(fig)
# fig.show()
fig.write_html("challenge2.html", auto_open=True)

# How to see all markers available
# https://plotly.com/python/marker-style/
raw_symbols = plotly.validators.scatter.marker.SymbolValidator().values
namestems = []
namevariants = []
symbols = []
for i in range(0, len(raw_symbols), 3):
    name = str(raw_symbols[i + 2])
    symbols.append(raw_symbols[i])
    namestems.append(name.replace("-open", "").replace("-dot", ""))
    namevariants.append(name[len(namestems[-1]):])

fig = go.Figure(go.Scatter(mode="markers", x=namevariants, y=namestems, marker_symbol=symbols,
                           marker_line_color="midnightblue", marker_color="lightskyblue",
                           marker_line_width=2, marker_size=15,
                           hovertemplate="name: %{y}%{x}<br>number: %{marker.symbol}<extra></extra>"))
fig.update_layout(title="Mouse over symbols for name & number!",
                  xaxis_range=[-1, 4], yaxis_range=[len(set(namestems)), -1],
                  margin=dict(b=0, r=0), xaxis_side="top", height=1200, width=400)
# fig.show()
fig.write_html("demo3.html", auto_open=True)

# plotly express is higher level api designed for data exploration
df1 = pd.DataFrame(data={"id": [1, 2, 3], "score": [1, 3, 2], "group": ["up-down"] * 3})
df2 = pd.DataFrame(data={"id": [1, 2, 4], "score": [1, 2, 5], "group": ["climber"] * 3})
df = pd.concat([df1, df2])
print(df)
px.line(df, x="id", y="score", color="group")

# Challenge 3: Using x and y from Challenge 1 plot the values using plotly express

# Solution to challenge 3
df = pd.DataFrame(data=y, index=x, columns=["sigmoid"])
df['fn'] = 'sigmoid'
print(f"Challenge 3 df.head()\n{df.head()}")
# px.line and px.scatter use dataframe index on x axis by default
fig = px.line(df, y="sigmoid", title="Sigmoid function constructed from plotly express", color='fn')
# fig  # same as fig.show() for colab and jupyter notebooks
fig.write_html("challenge3.html", auto_open=True)

# plotly express comes with some default dataframes if you want to practise
df = px.data.gapminder().query("continent=='Oceania'")
fig = px.line(df, x="year", y="lifeExp", color='country')
# fig.show()  # colab and jupyter notebook
fig.write_html("demo4.html", auto_open=True)

# Load data from the covid-19 data set. NB URL has been changed on Github. Now ends in "_global"
df_confirmed = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
    'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
print(f"df_confirmed.head() as read from github \n{df_confirmed.head()}")

# List the states in alphabetical order. Need to drop NaN because otherwise sort will break
ar_state = df_confirmed['Province/State'].dropna().unique()
ar_state.sort()
print(f"type(ar_state) {type(ar_state)}")
print(ar_state)

# Challenge 4: Print the list of countries in alphabetical order with no repeats

# Solution to challenge 4: Don't need to dropna because there are none. No harm if left in apart from slightly slower
ar_country = df_confirmed['Country/Region'].unique()
ar_country.sort()
print(f"Challenge 4 {ar_country}")

# how to filter by country
print("how to filter by country")
print(df_confirmed[df_confirmed["Country/Region"] == "Australia"])

# how to filter by state
print("how to filter by state")
print(df_confirmed[df_confirmed["Province/State"] == "Queensland"])

# how to filter by state and country should two countries have state with same name
print("how to filter by state and country")
print(df_confirmed[(df_confirmed["Province/State"] == "Queensland") & (df_confirmed["Country/Region"] == "Australia")])


# Challenge 5: define a function which takes 3 arguments, df, country, state and
# will return the dataframe filtered by country, by state or by country and state


# Solution to challenge 5
def df_for_location(df, country=None, state=None):
    filt = [True] * df.shape[0]
    if country:
        filt = filt & (df["Country/Region"] == country)
    if state:
        filt = filt & (df["Province/State"] == state)
    return df[filt]


# check it works
print("Challenge 5")
print(df_for_location(df_confirmed, country="Australia"))

df_confirmed_by_location = df_for_location(df_confirmed, country="Australia")

# Sum the rows in the dataframe and return a series
total = df_confirmed_by_location.sum(axis="index")
print(f"DataFrame.sum: type(total) {type(total)}")
print(f"DataFrame.sum result:\n{total}")

# Don't want the first four rows so can slice
series_sum = total[4:]
print(f"series_sum {series_sum}")

# index is currently str but would prefer datetime
print(f"series_sum.index before: {series_sum.index}")
series_sum.index = pd.to_datetime(series_sum.index)
print(f"series_sum.index after : {series_sum.index}")

# Challenge 6: define a function which returns a series of values for a given df and country and/or state
# Index for series should be a DateTimeIndex


# Solution to challenge 6
def series_sum_for_location(df, country=None, state=None):
    df = df_for_location(df=df, state=state, country=country)
    series = df.sum(axis="index")[4:].astype(int)
    series.index = pd.to_datetime(series.index)
    return series


# check it works
print("Challenge 6")
print(series_sum_for_location(df_confirmed, country="Australia"))

# Challenge 7: Define a function location_name which takes country and/or state and
# returns a name for that location
# location_name(country="Australia") should return "Australia"
# location_name(state="Queensland") should return "Queensland"
# location_name() should return "everywhere"
# location_name(country="Australia", state="Queensland") should return "Queensland - Australia"


# Solution to challenge 7
def location_name(country=None, state=None):
    locations = []
    if state:
        locations.append(state)
    if country:
        locations.append(country)
    return " - ".join(locations) if len(locations) > 0 else "everywhere"


# check it works
print('Challenge 7')
print(f'location_name(country="Australia", state="Queensland") {location_name(country="Australia", state="Queensland")}')
print(f'location_name(country="Australia") {location_name(country="Australia")}')
print(f'location_name(state="Queensland") {location_name(state="Queensland")}')
print(f'location_name() {location_name()}')

# To get the index for values greater than a starting value (in this case 100) use a filter
series_sum = series_sum_for_location(df_confirmed, country="Australia")
print("Index where value >= 100")
print(series_sum.index[series_sum >= 100])

# To convert to dataframe and label the data column 'current'
print("Convert to DataFrame")
df_sum = pd.DataFrame(series_sum, columns=['current'])
print(df_sum)

# Challenge 8: Find the date the number of confirmed cases exceeded 100 and what the count was on that day


# Solution to Challenge 8
idx_start = series_sum.index[series_sum >= 100][0]
num_start_actual = series_sum[idx_start]
# idx_start = df_sum.index[df_sum['current']>=100][0]
# num_start_actual = df_sum.loc[idx_start, 'current']
print("Challenge 8", idx_start, num_start_actual)

# Slice the Dataframe to only include those records greater than 100
df_plot = df_sum.loc[idx_start:]
print("Sliced df_plot")
print(df_plot)


# Challenge 9: Write a function to plot those numbers after starting count reached. Default starting count=100
# Advanced: plot count on a logarithmic scale


# Solution to challenge 9:
def plot_for_location(df, country=None, state=None, num_start=100, description=""):
    series_sum = series_sum_for_location(df, country=country, state=state)
    idx_start = series_sum.index[series_sum >= num_start][0]
    df_sum = pd.DataFrame(series_sum, columns=['current'])
    df_plot = df_sum.loc[idx_start:]
    location = location_name(country=country, state=state)
    fig = go.Figure(
        data=[go.Scatter(x=df_plot.index, y=df_plot['current'], name=location, showlegend=True, mode="lines")],
        layout=go.Layout(
            title=go.layout.Title(text=f"{description} cases for {location} starting from {num_start}"),
        )
    )
    fig.update_yaxes(type='log')
    return fig


# plot_for_location(df_confirmed, description="Confirmed")  # colab and jupyter notebook
plot_for_location(df_confirmed, description="Confirmed").write_html("challenge9.html", auto_open=True)

# Calculate the average doubling rate over the last 3 days
averaged_days = 3
idx_start = series_sum.index[series_sum >= 100][0]
df_sum = pd.DataFrame(series_sum, columns=['current'])
df_sum['previous'] = df_sum['current'].shift(averaged_days, fill_value=0)
df_plot = df_sum.loc[idx_start:].copy()  # need .copy() or will get warning later on
print("shifted results")
print(df_plot)

# Calculate how many days to double
df_plot['doubling days'] = averaged_days / np.log2(df_plot['current'] / df_plot['previous'])
print("calculated doubling days")
print(df_plot)


# Challenge 10: Plot days to double
# Advanced: Plot at same time as count using plotly subplots


# Solution to challenge 10
def plot_doubling_for_location(df, country=None, state=None, num_start=100, description="", averaged_days=3):
    series_sum = series_sum_for_location(df, country=country, state=state)
    idx_start = series_sum.index[series_sum >= num_start][0]
    df_sum = pd.DataFrame(series_sum, columns=['current'])
    df_sum['previous'] = df_sum['current'].shift(averaged_days, fill_value=0)
    df_plot = df_sum.loc[idx_start:].copy()
    df_plot['doubling days'] = averaged_days / np.log2(df_plot['current'] / df_plot['previous'])
    location = location_name(country=country, state=state)
    fig = go.Figure(
        data=[go.Scatter(x=df_plot.index, y=df_plot['doubling days'], name=location, showlegend=True, mode="lines")],
        layout=go.Layout(
            title=go.layout.Title(text=f"Days to double averaged over last {averaged_days} days. Higher is better")
        )
    )
    return fig


# plot_doubling_for_location(df_confirmed, country="Australia", description="Confirmed")  # colab and jupyter notebook
plot_doubling_for_location(df_confirmed, "Australia", description="Confirmed").write_html("challenge10.html", auto_open=True)


# Advanced solution to challenge 10
def plot_for_location(df, country=None, state=None, num_start=100, description="", averaged_days=3):
    series_sum = series_sum_for_location(df, country=country, state=state)
    idx_start = series_sum.index[series_sum >= num_start][0]
    df_sum = pd.DataFrame(series_sum, columns=['current'])
    df_sum['previous'] = df_sum['current'].shift(averaged_days, fill_value=0)
    df_plot = df_sum.loc[idx_start:].copy()
    df_plot['doubling days'] = averaged_days / np.log2(df_plot['current'] / df_plot['previous'])
    location = location_name(country=country, state=state)
    fig = plotly.subplots.make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        specs=[[{"rowspan": 2}], [None], [{}]],
        subplot_titles=["Confirmed cases on a logarithmic scale",
                        f"Days to double averaged over last {averaged_days} days. Higher is better"]
    )
    fig.update_layout(
        title_text=f"{description} cases {location} starting from {num_start}",
        height=600
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['current'], mode='lines', name=location),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['doubling days'], mode='lines', showlegend=False),
        row=3, col=1
    )
    fig.update_yaxes(title_text='Cases', type='log', row=1, col=1)
    idx_end = df_plot.index[-1]
    duration = (idx_end - idx_start).days
    doubler = 6  # draw a line for doubling every 6 days
    num_start_actual = df_plot.loc[idx_start, 'current']
    num_end = int(num_start_actual * 2 ** (duration / doubler))
    fig.add_trace(
        go.Scatter(x=[idx_start, idx_end], y=[num_start_actual, num_end], mode='lines', name=f'every {doubler} days'),
        row=1, col=1
    )
    return fig


# plot_for_location(df_confirmed, country="Australia", description="Confirmed")   # colab
plot_for_location(df_confirmed, country="Australia", description="Confirmed").write_html("challenge10adv.html", auto_open=True)


# Challenge 11: Add lines on log chart for doubling every 2, 3, 4, 5 days


# Solution to challenge 11
def plot_for_location(df, country=None, state=None, num_start=100, description="", averaged_days=3):
    series_sum = series_sum_for_location(df, country=country, state=state)
    idx_start = series_sum.index[series_sum >= num_start][0]
    df_sum = pd.DataFrame(series_sum, columns=['current'])
    df_sum['previous'] = df_sum['current'].shift(averaged_days, fill_value=0)
    df_plot = df_sum.loc[idx_start:].copy()
    df_plot['doubling days'] = averaged_days / np.log2(df_plot['current'] / df_plot['previous'])
    location = location_name(country=country, state=state)
    fig = plotly.subplots.make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        specs=[[{"rowspan": 2}], [None], [{}]],
        subplot_titles=["Confirmed cases on a logarithmic scale",
                        f"Days to double averaged over last {averaged_days} days. Higher is better"]
    )
    fig.update_layout(
        title_text=f"{description} cases {location} starting from {num_start}",
        height=600
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['current'], mode='lines', name=location),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['doubling days'], mode='lines', showlegend=False),
        row=3, col=1
    )
    fig.update_yaxes(title_text='Cases', type='log', row=1, col=1)
    idx_end = df_plot.index[-1]
    duration = (idx_end - idx_start).days
    for doubler in (2, 3, 4, 5, 6):
        num_start_actual = df_plot.loc[idx_start, 'current']
        num_end = int(num_start_actual * 2 ** (duration / doubler))
        fig.add_trace(
            go.Scatter(x=[idx_start, idx_end], y=[num_start_actual, num_end], mode='lines',
                       name=f'every {doubler} days'),
            row=1, col=1
        )
    return fig


# plot_for_location(df_confirmed, country="Australia", description="Confirmed")  # colab
plot_for_location(df_confirmed, country="Australia", description="Confirmed").write_html("challenge11.html", auto_open=True)
