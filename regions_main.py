# -*- coding: utf-8 -*-

# IMPORTS

# for analysis
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime as dt

# for plotly plotting
import plotly.io as pio
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objects as go

import region_utils

import timeit
from sklearn.mixture import GaussianMixture

#%%

# CONSTANTS THAT MUST BE SET BY USER

# folder where all data is saved
dirname = "SET_DIRECTORY_NAME"

# token for mapbox plotting
# https://docs.mapbox.com/help/getting-started/access-tokens/
mapbox_token = "SET_MAPBOX_TOKEN"


#%% SET UP CONSTANTS

bbox_3857 = [
    12876276.0742,
    -3810385.8286,
    12928711.7845,
    -3718508.5162,
]  # 3857. unused for regions
bbox_4326 = [115.669556, -32.357923, 116.140594, -31.658057]  # 4326. unused for regions
small_bbox_4326 = [115.734100, -32.005747, 116.024551, -31.872310]  # unused for regions

# EPSG codes for common CRS: 4326 = WGS84 from GPS, 3857 = web mercator, GDA2020 = 7844

# For Plotly plotting
pio.renderers.default = "browser"
pyo.offline.init_notebook_mode()
px.set_mapbox_access_token(mapbox_token)

# Colour pallette. Change here as necessary.
mycolors_discrete = [
    "#324472",
    "#2693A1",
    "#EFA517",
    "#E86D1F",
    "#A41410",
    "#323232",
    "#E4E0A9",
    "#D2CAB5",
]
mycolors_continuous = [
    "#EFA517",
    "#E86D1F",
    "#A41410",
    "#2693A1",
    "#324472",
    "#323232",
]  # for continuous gradient

# distance in km between origin and final stops
# to count as a return trip (rather than onwards)
RETURN_THRESHOLD = 0.2

# for anchoring regions
STAY_DROP_THRESHOLD = 0
MIN_POINTS_REGION = 1
MAPBUFFER = 20  # really just for mapping, buffer the anchor stops for the convex hull
STOPBUFFER = 400  # distance in m to buffer stops by for the land use
ANCHOR_REGION_SIZE = 2 * STOPBUFFER  # distance between stops for regions
MAX_TIME = 0.8  # % of time in the month for a region to be called an 'anchor' region

# stay classes
classmap = {
    1: "W",
    2: "E",
    3: "S",
    4: "M",
    5: "C",
    6: "C",
    7: "C",
    8: "C",
    9: "L",
    10: "V",
    12: "T",
    0: "drop",
}
# 1 work (W)
# 2 school (E education)
# 3 short (S)
# 4 medium (M)
# 5 AM transfer (C change)
# 6 day transfer (C change)
# 7 PM transfer (C change)
# 8 overnight transfer (C change)
# 9 sleep (L long)
# 10 very long (V)
# 12 travel (T)


#%% IMPORT JOURNEY DATA

journeys = pd.read_csv(
    dirname + "journeys.csv",
    usecols=[
        "Cardid",
        "OnDate",
        "OnLocation",
        "OnMode",
        "EtimCode",
        "OffDate",
        "OnTran",
        "OffTran",
        "OffLocation",
        "Token",
    ],
)

#%% HISTOGRAM OF NUMBER OF CARD USES - BEFORE ANY TRIPS DROPPED

# Figure 1 in readme

v = journeys["Cardid"].value_counts()
x = v[v.lt(100)]  # to plot a zoomed histogram

fig = px.histogram(
    x,
    color_discrete_sequence=mycolors_discrete,
    labels={"value": "Number of card uses"},
)
fig.update_layout(
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor="rgb(243, 243, 243)",
        gridwidth=1,
        zerolinecolor="rgb(243, 243, 243)",
        zerolinewidth=2,
        separatethousands=True,
        tickformat=",",
    ),
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgb(255, 255, 255)",
)
pyo.plot(fig)


# how many times does a card have to be used in a month for it to be kept?
# determined from this chart
MIN_USES = 12

#%% READ PROCESSED JOURNEYS DATA

# Alternative to the next cell, read it in if already processed
journeys = pd.read_pickle(dirname + "processed_journeys.pkl")


#%% JOURNEYS DATA PROCESSING

journeys = region_utils.journey_processing(journeys, MIN_USES)
journeys.to_pickle(dirname + "processed_journeys.pkl")

#%% READ STOP CLUSTERS

# Alternative to the next cell - read in instead of running

clusters = pd.read_pickle(dirname + "clusters.pkl")
geo = pd.read_pickle(dirname + "geo.pkl")

#%% STOP DATA

busStops = pd.read_csv(dirname + "busStops.csv")
trainStops = pd.read_csv(dirname + "trainStops.csv")

# create geodataframe of both files
geo = region_utils.create_geo_stops(trainStops, busStops)

# work out which stops have been used
onstops = journeys["OnLocation"].value_counts()
onstops = pd.DataFrame(onstops)
onstops.reset_index(inplace=True)
onstops.rename(columns={"index": "StopID", "OnLocation": "Count On"}, inplace=True)

offstops = journeys["OffLocation"].value_counts()
offstops = pd.DataFrame(offstops)
offstops.reset_index(inplace=True)
offstops.rename(columns={"index": "StopID", "OffLocation": "Count Off"}, inplace=True)

geo = pd.merge(geo, onstops, left_on="StopID", right_on="StopID", how="left")

geo = pd.merge(geo, offstops, left_on="StopID", right_on="StopID", how="left")

geo["Count On"] = geo["Count On"].fillna(0)
geo["Count Off"] = geo["Count Off"].fillna(0)

geo["Count Total"] = geo["Count On"] + geo["Count Off"]

# remove stops that aren't used
geo = geo.drop(geo[geo["Count Total"] == 0].index).reset_index(drop=True)

#%% STOP CLUSTERING

# drop stops outside the bbox
# not really necessary as it's not a big list but leave here for later
# geo = geo.cx[
#    small_bbox_4326[0]:small_bbox_4326[2],
#    small_bbox_4326[1]:small_bbox_4326[3]
# ]

# DBSCAN - stop clustering
eps = 90  # metres
minpts = 1  # smallest cluster size allowed

# cluster stops
clusters, geo = region_utils.create_stop_cluster(geo, "spatial_cluster", eps, minpts)

geo.to_crs(epsg=4326, inplace=True)
geo["X"] = geo["geometry"].x
geo["Y"] = geo["geometry"].y

clusters.to_pickle(dirname + "clusters.pkl")
geo.to_pickle(dirname + "geo.pkl")

#%% TEST PLOT OF CLUSTERS

# Plot of all stops coloured by spatial cluster
# Demonstrates which stops have been aggregated together

# Figure 2 in readme

geopolys = geo.dissolve("spatial_cluster").convex_hull

geopolys = gpd.GeoDataFrame(geopolys)
geopolys.reset_index(inplace=True)
geopolys.rename(columns={0: "geometry"}, inplace=True)
geopolys = geopolys.set_crs("epsg:4326", allow_override=True)

geopolys = geopolys.to_crs(epsg=32749)
geopolys["geometry"] = geopolys["geometry"].buffer(MAPBUFFER)

geo.to_crs(epsg=4326, inplace=True)
geopolys.to_crs(epsg=4326, inplace=True)

fig = px.choropleth_mapbox(
    geopolys,
    geojson=geopolys.geometry,
    locations=geopolys.index,
    color="spatial_cluster",
    center={"lat": -32, "lon": 116},
    mapbox_style="open-street-map",
    color_continuous_scale=mycolors_discrete,
    opacity=0.5,
    zoom=10,
)
fig.update_layout(mapbox_style="light")

fig.add_trace(
    go.Scattermapbox(
        lat=geo["geometry"].y,
        lon=geo["geometry"].x,
        mode="markers",
        marker=go.scattermapbox.Marker(
            color=geo["spatial_cluster"],
            colorscale=mycolors_discrete,
        ),
    )
)

pyo.plot(fig)


#%% READ ACTIVITIES FILE

# Alternative to the next two cells - read in instead of running

activities = pd.read_pickle(dirname + "activities.pkl")


#%% ANCHORING REGIONS - SETUP

start_date = "2017-08-01 00:00:00"
start_date = pd.to_datetime(start_date, format="%Y-%m-%d %H:%M:%S")

# calculate total hours since start of the month
journeys["elapsedHours_On"] = (journeys["OnTime"] - start_date) / dt.timedelta(hours=1)

journeys["elapsedHours_Off"] = (journeys["OffTime"] - start_date) / dt.timedelta(
    hours=1
)

#%% TURN JOURNEYS INTO STAYS

# get sequence of activities by card
activities = pd.melt(
    journeys,
    id_vars=["Cardid", "Token", "OnLocation", "OffLocation", "EtimCode"],
    value_vars=["elapsedHours_On", "elapsedHours_Off"],
)
# put back in order
activities.sort_values(by=["Cardid", "value"], axis=0, ascending=True, inplace=True)
# simplify variable names
activities["location"] = np.where(
    activities["variable"] == "elapsedHours_On",
    activities["OnLocation"],
    activities["OffLocation"],
)
activities["variable"] = np.where(
    activities["variable"] == "elapsedHours_On", "On", "Off"
)

activities["Duration"] = (
    activities["value"]
    .shift(-1)
    .where(activities["Cardid"].shift(-1) == activities["Cardid"])
    - activities["value"]
)
# separate into travel vs stay
activities["activity"] = np.where(activities["variable"] == "On", "T", "stay")

# get coordinates of locations
activities = pd.merge(
    activities, geo, left_on="location", right_on="StopID", how="left"
)
activities.rename(columns={"X": "LocationLon", "Y": "LocationLat"}, inplace=True)

# use this to get the *next* spatial cluster
activities = pd.merge(
    activities, geo, left_on="OffLocation", right_on="StopID", how="left"
)

activities.drop(
    ["StopID_x", "StopID_y", "X", "Y", "geometry_x", "geometry_y"], axis=1, inplace=True
)

activities.rename(
    columns={
        "spatial_cluster_x": "spatial_cluster",
        "spatial_cluster_y": "spatial_cluster_next",
    },
    inplace=True,
)

# calculate distance travelled between start and end locations of the stay
# approximation only - ok for smaller distances
activities["Distance_stay"] = 110.3 * np.sqrt(
    np.square(activities["LocationLon"] - activities["LocationLon"].shift(-1))
    + np.square(activities["LocationLat"] - activities["LocationLat"].shift(-1))
).where(activities["Cardid"] == activities["Cardid"].shift(-1))

activities["Start_h"] = activities["value"]
activities["Start_h_only"] = activities["value"].mod(24)

activities["End_h"] = activities["Start_h"] + activities["Duration"]
activities.rename(columns={"value": "Cumulative_h"}, inplace=True)

activities["Token"], activities["class"] = region_utils.learn_stay_class(
    activities, "Duration", "Start_h_only", "Token", "Distance_stay", RETURN_THRESHOLD
)
activities["stayclass"] = activities["class"].map(classmap)
activities["activity_alpha"] = np.where(
    activities["activity"] == "stay", activities["stayclass"], activities["activity"]
)
activities["activity_int"] = np.where(
    activities["activity"] == "stay", activities["class"], 12
)  # 12 is the class for travel in the classmap

# consolidate same classes
agg_classmap = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 5,
    7: 5,
    8: 5,
    9: 6,
    10: 7,
    11: 6,
    12: 8,
    0: 0,
}
# remap to alpha
agg_classmap_a = {
    1: "W",
    2: "E",
    3: "S",
    4: "M",
    5: "C",
    6: "L",
    7: "R",
    8: "T",
    0: "drop",
}

activities["activity_agg"] = activities["activity_int"].map(agg_classmap)
activities["activity_agg_alpha"] = activities["activity_agg"].map(agg_classmap_a)

activities = activities.drop(
    activities[activities["activity_agg_alpha"] == "drop"].index
).reset_index(drop=True)

# drop all travel (T) and transfer (C) activities
activities = activities.drop(
    activities[
        (activities["activity_agg_alpha"] == "T")
        | (activities["activity_agg_alpha"] == "C")
    ].index
).reset_index(drop=True)

activities.drop(
    [
        "stayclass",
        "class",
        "activity",
        "variable",
        "OnLocation",
        "OffLocation",
        "Start_h_only",
    ],
    axis=1,
    inplace=True,
)

activities.to_pickle(dirname + "activities.pkl")

#%% HISTOGRAM OF DISTANCES IN STAYS

# For exploration only. Distance between start and end locations of each stay
# Used to confirm assumption that the land use around the stay location can be
# used to infer activity. If the distance is large, the passenger does not stay 'stay'
# in the location and thus the land use doesn't inform the activity type.

# Figure 3 in readme

# Long tail, so truncate at this value for plotting
x = 10

fig = px.histogram(
    activities["Distance_stay"][activities["Distance_stay"] < x],
    color_discrete_sequence=mycolors_discrete,
    barmode="group",
    labels={"value": "Stay distance (km)"},
)
fig.update_layout(
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor="rgb(243, 243, 243)",
        gridwidth=1,
        zerolinecolor="rgb(243, 243, 243)",
        zerolinewidth=2,
        separatethousands=True,
        tickformat=",",
    ),
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgb(255, 255, 255)",
)
pyo.plot(fig)

#%% READ CARD SUMMARY

cards = pd.read_pickle(dirname + "cards.pkl")

#%% GENERATE CARD SUMMARY

# summarise cards to make it easier to chart/investigate/choose what to look at
cards = activities["Cardid"].value_counts()
cards = pd.DataFrame(cards)
cards.reset_index(inplace=True)
cards.rename(columns={"Cardid": "count", "index": "Card"}, inplace=True)

print(cards[(cards["count"] > 100) & (cards["count"] < 150)])

cards.to_pickle(dirname + "cards.pkl")

#%% LAND USE DATA

mbpoly = gpd.read_file(dirname + "MB_2016_WA.shp")
mbpoly = mbpoly.to_crs(epsg=4326)

# create mask for land use categories
landmap = {
    "Residential": "Residential",
    "Commercial": "Industrial/Commercial",
    "Industrial": "Industrial/Commercial",
    "Transport": "Industrial/Commercial",
    "Primary Production": "Industrial/Commercial",
    "Other": "Industrial/Commercial",
    "Parkland": "Parks/Water",
    "Water": "Parks/Water",
    "Education": "Education",
    "Hospital/Medical": "Hospital/Medical",
}

mbpoly["MB_CAT16"] = mbpoly["MB_CAT16"].map(landmap)

mbdissolve = mbpoly.dissolve(by="MB_CAT16")
mbdissolve.reset_index(inplace=True)

#%% CREATE RANDOM SAMPLE OF CARDS

np.random.seed(42)

# allocate each card to a number between 1 and 5
cards["split"] = np.random.randint(1, 6, cards.shape[0])

cardrun = cards[cards["split"] == 4]
cardrun.reset_index(inplace=True)

#%% READ REGIONS

allhist = pd.read_pickle(dirname + "fullallhist.pkl")
allregions = pd.read_pickle(dirname + "fullallregions.pkl")
allregionpolys = pd.read_pickle(dirname + "fullallregionpolys.pkl")

#%% GENERATE REGIONS

# This takes about a minute per 250 cards (only approximate - varies with the 
# amount of activities on each card; more activities = takes longer)

start = timeit.default_timer()

allhist = pd.DataFrame()
allregions = pd.DataFrame()
allregionpolys = pd.DataFrame()

for index, row in cardrun.iterrows():
    if np.remainder(index, 50) == 0:
        print(
            str(index)
            + " of "
            + str(len(cardrun))
            + " cards complete, "
            + str(timeit.default_timer() - start)
            + " seconds elapsed"
        )
    ret = region_utils.generate_regions(
        row["Card"],
        clusters,
        activities,
        STAY_DROP_THRESHOLD,
        ANCHOR_REGION_SIZE,
        MIN_POINTS_REGION,
        MAPBUFFER,
    )

    if ret is None:
        continue
    hist, region_summary, regionpolys = ret
    hist["Card"] = row["Card"]
    region_summary["Card"] = row["Card"]
    regionpolys["Card"] = row["Card"]

    allhist = allhist.append(hist)
    allregions = allregions.append(region_summary)
    allregionpolys = allregionpolys.append(regionpolys)
print("All cards complete, " + str(timeit.default_timer() - start) + " seconds elapsed")

# Export so have a full list of regions (not just the anchoring ones)

allhist.to_pickle(dirname + "fullallhist.pkl")
allregions.to_pickle(dirname + "fullallregions.pkl")
allregionpolys.to_pickle(dirname + "fullallregionpolys.pkl")

#%% HISTOGRAM OF NUMBER OF VISITS TO EACH REGION

# Used to assess for an 'elbow' - is there a number of vists at which there is
# a clear inflection?

# Figure 4 in readme

# Long tail - truncate for graphing only to make it easier to see

fig = px.histogram(
    allregions["num_visits"][allregions["num_visits"] < 20],
    color_discrete_sequence=mycolors_discrete,
    labels={
        "value": "Number of visits to region",
        "count": "Cards",
    },
)
fig.update_layout(
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor="rgb(243, 243, 243)",
        gridwidth=1,
        zerolinecolor="rgb(243, 243, 243)",
        zerolinewidth=2,
    ),
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgb(255, 255, 255)",
)
pyo.plot(fig)

# From this plot, determine elbow at 5 regions
# Any region with >= 5 visits is called an 'anchor' region
# This works out to a region with a visit on average at least approximately
# once per week - so makes intuitive sense too
NUM_REGION_VISITS = 5

#%% DETERMINE ANCHORING REGIONS

allregions["region_type"] = np.where(
    allregions["num_visits"] >= NUM_REGION_VISITS, "Anchor", "Visited"
)

allhist = pd.merge(
    allhist,
    allregions[["Card", "region_cluster", "region_type"]],
    left_on=["Card", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)
allregionpolys = pd.merge(
    allregionpolys,
    allregions[["Card", "region_cluster", "region_type"]],
    left_on=["Card", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)

#%% COVERAGE OF ANCHORING REGIONS

regionsummary = pd.pivot_table(
    allregions,
    values=["fraction_time", "fraction_visits"],
    index=["Card"],
    columns="region_type",
    fill_value=0,
    aggfunc=[np.sum, "count"],
)
regionsummary.reset_index(inplace=True)

timesummary = regionsummary.loc[:, ("sum", "fraction_time", "Anchor")]
d = {"fraction_time": timesummary}
timesummary = pd.DataFrame(data=d)

visitsummary = regionsummary["sum", "fraction_visits", "Anchor"]
d = {"fraction_visits": visitsummary}
visitsummary = pd.DataFrame(data=d)

print(
    len(timesummary[timesummary["fraction_time"] > 0.8]) / 
    len(timesummary[timesummary["fraction_time"] > 0])
)  
# ~72% of cards have that have anchoring regions, have anchoring regions 
# covering more than 80% of the month

print(
    len(visitsummary[visitsummary["fraction_visits"] > 0.8]) / 
    len(visitsummary[visitsummary["fraction_visits"] > 0])
)  
# ~70% of cards that have anchoring regions, have anchoring regions 
# covering more than 80% of their visits

#%% HISTOGRAM - FRACTION OF TIME COVERED BY ANCHORING REGIONS

# Figure 5a in readme

fig = px.histogram(
    timesummary,
    color_discrete_sequence=mycolors_discrete,
    labels={
        "value": "Fraction of time covered by anchoring regions",
        "count": "Cards",
    },
)
fig.update_layout(
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor="rgb(243, 243, 243)",
        gridwidth=1,
        zerolinecolor="rgb(243, 243, 243)",
        zerolinewidth=2,
    ),
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgb(255, 255, 255)",
)
pyo.plot(fig)

#%% HISTOGRAM - FRACTION OF VISITS COVERED BY ANCHORING REGIONS

# Figure 5b in readme

fig = px.histogram(
    visitsummary,
    color_discrete_sequence=mycolors_discrete,
    labels={
        "value": "Fraction of visits covered by anchoring regions",
        "count": "Cards",
    },
)
fig.update_layout(
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor="rgb(243, 243, 243)",
        gridwidth=1,
        zerolinecolor="rgb(243, 243, 243)",
        zerolinewidth=2,
    ),
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgb(255, 255, 255)",
)
pyo.plot(fig)

#%% SEPARATE OUT ANCHORING REGIONS ONLY

allhist = allhist[allhist["region_type"] == "Anchor"]
allregions = allregions[allregions["region_type"] == "Anchor"]
allregionpolys = allregionpolys[allregionpolys["region_type"] == "Anchor"]


#%% READ FILES

# instead of the next cell - import instead

allregions = pd.read_pickle(dirname + "final-allregions.pkl")
landusepolys = pd.read_pickle(dirname + "final-landusepolys.pkl")
regionpivot = pd.read_pickle(dirname + "final-regionpivot.pkl")


#%% OVERLAY LAND USE ON REGIONS

# This also takes a while - the same time if not longer than the region generation

start = timeit.default_timer()

# Buffer the stops by the STOPBUFFER (circles around the stops)
landusepolys = allhist.to_crs(epsg=32749)
landusepolys["geometry"] = landusepolys["geometry"].buffer(STOPBUFFER)

# Concatenate the regional polygons and the buffered stops
landusepolys = pd.concat([landusepolys, allregionpolys], ignore_index=True, sort=False)
print(
    "Geodataframes concatenated, "
    + str(timeit.default_timer() - start)
    + " seconds elapsed"
)

# Dissolve by card and region_cluster
landusepolys = landusepolys.dissolve(by=["Card", "region_cluster"])
landusepolys.reset_index(inplace=True)

landusepolys["area"] = landusepolys.area
landusepolys.sort_values(by=["area"], axis=0, ascending=False, inplace=True)

print("Regions dissolved, " + str(timeit.default_timer() - start) + " seconds elapsed")

# Get extent of the polygons and reduce the land use dataset to those bounds
landusepolys = landusepolys.to_crs(epsg=4326)
xmin, ymin, xmax, ymax = landusepolys.total_bounds
mbdissolve_small = mbdissolve.cx[xmin:xmax, ymin:ymax]

print(
    "Land use bounds reduced, "
    + str(timeit.default_timer() - start)
    + " seconds elapsed"
)

# Make sure in same CRS
mbdissolve_small = mbdissolve_small.to_crs(epsg=32749)
landusepolys = landusepolys.to_crs(epsg=32749)

# Overlay land use - this is the slow part
ABSoverlay = mbdissolve_small.overlay(landusepolys, how="intersection")
ABSoverlay["area"] = ABSoverlay["geometry"].area

print("Land use overlayed, " + str(timeit.default_timer() - start) + " seconds elapsed")

# Calculate area of each land use type for each region cluster
ABSpivot = pd.pivot_table(
    ABSoverlay,
    values="area",
    index=["Card", "region_cluster"],
    columns="MB_CAT16",
    fill_value=0,
    aggfunc=np.sum,
)
ABSpivot.reset_index(inplace=True)
ABSpivot["Total"] = ABSpivot.iloc[:, 2:7].sum(axis=1)

print("Areas calculated, " + str(timeit.default_timer() - start) + " seconds elapsed")

# Turn these areas into fractions
ABSpivot.iloc[:, 2:7] = ABSpivot.iloc[:, 2:7].div(ABSpivot["Total"], axis=0)
ABSpivot.drop("Total", inplace=True, axis=1)

print(
    "Fractions calculated, " + str(timeit.default_timer() - start) + " seconds elapsed"
)
#%%
activityregions = pd.merge(
    activities,
    allhist[["Card", "spatial_cluster", "region_cluster", "region_type"]],
    left_on=["Cardid", "spatial_cluster"],
    right_on=["Card", "spatial_cluster"],
    how="left",
)
activityregions = activityregions.dropna(
    axis="rows"
)  # because it hasn't been run for all cards, most will be NaN

regionpivot = pd.pivot_table(
    activityregions,
    index=["Cardid", "region_cluster", "region_type"],
    columns="activity_alpha",
    aggfunc="count",
    values="Card",
    fill_value=0,
)
regionpivot.reset_index(inplace=True)

regionpivot["Total"] = regionpivot.iloc[:, 3:9].sum(
    axis=1
)  # includes V = very long stays (>20h)
regionpivot["Total"] = regionpivot["Total"] - regionpivot["V"]  
# subtract off the V

# Drop regions that only have the very long stays - these will be where 
# the total now equals 0
regionpivot.drop(
    regionpivot[regionpivot["Total"] == 0].index, inplace=True
)


# This dataset includes all regions, so drop the non-anchoring ones
regionpivot.drop(
    regionpivot[regionpivot["region_type"] == "Visited"].index, inplace=True
)

regionpivot["E_frac"] = regionpivot["E"].div(regionpivot["Total"], axis=0)
regionpivot["L_frac"] = regionpivot["L"].div(regionpivot["Total"], axis=0)
regionpivot["M_frac"] = regionpivot["M"].div(regionpivot["Total"], axis=0)
regionpivot["S_frac"] = regionpivot["S"].div(regionpivot["Total"], axis=0)
regionpivot["W_frac"] = regionpivot["W"].div(regionpivot["Total"], axis=0)

regionpivot["regionid"] = (
    regionpivot["Cardid"].astype(str) + "-" 
    + regionpivot["region_cluster"].astype(str)
)
regionpivot = pd.merge(
    regionpivot,
    ABSpivot,
    left_on=["Cardid", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)

allregions.to_pickle(dirname + "final-allregions.pkl")
landusepolys.to_pickle(dirname + "final-landusepolys.pkl")
regionpivot.to_pickle(dirname + "final-regionpivot.pkl")

#%% CLUSTER REGIONS - GAUSSIAN MIXTURE MODEL

X = regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]]

# To determine the appropriate amount of clusters
# Run GMM across 1 to 20 clusters and check AIC/BIC
n_components = np.arange(1, 20)
models = [GaussianMixture(n, random_state=0).fit(X) for n in n_components]

d = {"BIC": [m.bic(X) for m in models], "AIC": [m.aic(X) for m in models]}
df = pd.DataFrame(data=d)

# Figure 6 in readme

fig = px.line(
    df,
    color_discrete_sequence=mycolors_discrete,
    labels={"index": "Number of clusters", 
            "value": "", 
            "variable": "Criteria"},
)
fig.update_layout(
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor="rgb(243, 243, 243)",
        gridwidth=1,
        zerolinecolor="rgb(243, 243, 243)",
        zerolinewidth=2,
    ),
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgb(255, 255, 255)",
)
pyo.plot(fig)

# From this plot, determine that the appropriate number of clusters
NUM_COMPONENTS = 6

#%% FIT GMM TO SELECTED NUMBER OF CLUSTERS

GMM = GaussianMixture(n_components=NUM_COMPONENTS, random_state=0).fit(X)
colname = "GMM_cluster"

regionpivot[colname] = GMM.predict(
    regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]]
)

# Add point geometries
regionpivotpoint = pd.merge(
    regionpivot,
    allregions,
    left_on=["Cardid", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)
regionpivotpoint = gpd.GeoDataFrame(regionpivotpoint, geometry="geometry")

# Add polygon geometries
regionpivotpoly = pd.merge(
    regionpivot,
    landusepolys,
    left_on=["Cardid", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)

outregion = pd.melt(
    regionpivot,
    id_vars=[colname],
    value_vars=["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"],
)
outland = pd.melt(
    regionpivot,
    id_vars=[colname],
    value_vars=[
        "Industrial/Commercial",
        "Education",
        "Hospital/Medical",
        "Parks/Water",
        "Residential",
    ],
)
outregionsum = pd.pivot_table(
    outregion, index=colname, columns="variable", aggfunc="sum"
)
outlandsum = pd.pivot_table(outland, index=colname, columns="variable", aggfunc="sum")

outland.sort_values(by=[colname, "variable"], axis=0, ascending=True, inplace=True)
outregion.sort_values(by=[colname, "variable"], axis=0, ascending=True, inplace=True)

#%% SUPPORTING OUTPUT CHARTS

numregions = pd.pivot_table(
    regionpivot, index="Cardid", columns="region_type", aggfunc="count", values=["Card"]
)

# Histogram of number of regions for each user
# Figure 7 in readme

fig = px.histogram(
    numregions.values,
    color_discrete_sequence=mycolors_discrete,
    labels={
        "value": "Number of regions per card",
        "count": "Number of cards",
    },
)
fig.update_layout(
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor="rgb(243, 243, 243)",
        gridwidth=1,
        zerolinecolor="rgb(243, 243, 243)",
        zerolinewidth=2,
        separatethousands=True,
        tickformat=",",
    ),
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgb(255, 255, 255)",
)
pyo.plot(fig)

#%%

# Histogram of number of regions per card by type

# Figure 8 in readme
cardsum = pd.pivot_table(
    regionpivot, index="Cardid", columns=colname, aggfunc="count", values=["Card"]
)
cardsum.fillna(0, inplace=True)

cardsum.columns = cardsum.columns.droplevel()

clustermap = {
    0: "Education",
    1: "Residences",
    2: "Workplaces/Leisure",
    3: "Education/Leisure",
    4: "Workplaces",
    5: "Leisure/Residences",
}

cardsum.rename(columns=clustermap, inplace=True)

cardsum["Total"] = cardsum.iloc[:, 0:6].sum(axis=1)

fig = px.histogram(
    cardsum["Total"][cardsum["Residences"] == 0], # change here for other types
    color_discrete_sequence=mycolors_discrete,
    barmode="group",
    labels={
        "value": "Number of 'Residence' regions per card",
        "count ": "Number of cards",
        colname: "Region cluster",
    },
)
fig.update_layout(
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor="rgb(243, 243, 243)",
        gridwidth=1,
        zerolinecolor="rgb(243, 243, 243)",
        zerolinewidth=2,
        separatethousands=True,
        tickformat=",",
    ),
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgb(255, 255, 255)",
)
pyo.plot(fig)

#%% BOXPLOT - ACTIVITY FRACTIONS BY REGION CLUSTER

# Figure 9 in readme

d = {"E_frac": "E", "L_frac": "L", "S_frac": "S", "M_frac": "M", "W_frac": "W"}

outregion["variable"] = outregion["variable"].map(d)

fig = px.box(
    outregion,
    x=colname,
    y="value",
    color="variable",
    color_discrete_sequence=mycolors_discrete,
    title=colname,
    labels={colname: "Region ID", "value": "Fraction", "variable": "Activity"},
)

fig.update_layout(
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor="rgb(243, 243, 243)",
        gridwidth=1,
        zerolinecolor="rgb(243, 243, 243)",
        zerolinewidth=2,
    ),
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgb(255, 255, 255)",
)

# fig.update_traces(boxpoints=False) # sets whiskers to min/max
fig.update_traces(
    marker=dict(opacity=0)
)  # sets whiskers to usual st dev and not 'outliers'
pyo.plot(fig)

#%% BOXPLOT - LAND USE FRACTIONS BY REGION CLUSTER

# Figure 10 in readme

fig = px.box(
    outland,
    x=colname,
    y="value",
    color="variable",
    color_discrete_sequence=mycolors_discrete,
    title=colname,
    labels={colname: "Region ID", "value": "Fraction", "variable": "Land use"},
)

fig.update_layout(
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor="rgb(243, 243, 243)",
        gridwidth=1,
        zerolinecolor="rgb(243, 243, 243)",
        zerolinewidth=2,
    ),
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgb(255, 255, 255)",
)

fig.update_traces(marker=dict(opacity=0))
pyo.plot(fig)

#%% PLOT REGION CENTROIDS BY CLUSTER TYPE

# This is to intuitively sense check the region types against a map with known
# features. (e.g. Is the CBD coming up as workplaces? Are the known shopping centres
# coming up as commercial? Needs to be plotted as continuous and not discrete

# Figure 11 in readme

regionpivotpoint = gpd.GeoDataFrame(regionpivotpoint, geometry="geometry")
regionpivotpoint.to_crs(epsg=4326, inplace=True)

regionpivotpoint["cluster_name"] = regionpivotpoint[colname].map(clustermap)

# suggest dropping residential to make it easier to see the others
plotregion = regionpivotpoint[regionpivotpoint["cluster_name"] != "Residences"]

fig = px.scatter_mapbox(
    plotregion,
    lat=plotregion["geometry"].y,
    lon=plotregion["geometry"].x,
    color=plotregion[colname],  
    color_discrete_sequence=mycolors_discrete,
    hover_name="regionid",
    hover_data=["E", "L", "M", "S", "W"],
    zoom=10,
)
fig.update_layout(mapbox_style="light")
pyo.plot(fig)

#%% plots for a selected card

testcard = 12897166 

# map coloured by region cluster
regionplot = regionpivotpoly[regionpivotpoly["Cardid"] == testcard]
regionplot[colname] = regionplot[colname].map(clustermap)

regionplot = gpd.GeoDataFrame(regionplot, geometry="geometry")
regionplot.to_crs(epsg=4326, inplace=True)

fig1 = px.choropleth_mapbox(
    regionplot,
    geojson=regionplot.geometry,
    locations=regionplot.index,
    color_discrete_sequence=mycolors_discrete,
    center={"lat": -32, "lon": 116},
    zoom=10,
    mapbox_style="light",
    opacity=0.5,
    hover_name="regionid",
    hover_data=["E", "L", "M", "S", "W"],
    color=regionplot[colname],
)
fig1.update_geos(fitbounds="locations", visible=False)

pyo.plot(fig1)

#%%

# bar chart of activities by region for selected card
# note that this isn't region type - the number that is the "region_cluster"
# is just the region ID for this card

test = pd.melt(
    regionplot, id_vars=["region_cluster"], value_vars=["E", "L", "M", "S", "W"]
)
testsum = pd.pivot_table(
    test, index="region_cluster", columns="variable", aggfunc="sum"
)  

fig = px.bar(
    test,
    x=test["region_cluster"].astype(str),
    y="value",
    color="variable",
    barmode="group",
    color_discrete_sequence=mycolors_discrete,
)
fig.update_layout(
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor="rgb(243, 243, 243)",
        gridwidth=1,
        zerolinecolor="rgb(243, 243, 243)",
        zerolinewidth=2,
    ),
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgb(255, 255, 255)",
)
pyo.plot(fig)


#%%

# map with regions broken out by land use for selected card

cardlanduse = ABSoverlay[ABSoverlay["Card"] == testcard]

cardlanduse.to_crs(epsg=4326, inplace=True)
cardlanduse.reset_index(inplace=True)

fig3 = px.choropleth_mapbox(
    cardlanduse,
    geojson=cardlanduse.geometry,
    locations=cardlanduse.index,
    center={"lat": -32, "lon": 116},
    zoom=10,
    mapbox_style="light",
    opacity=0.5,
    hover_name="Card",
    color="MB_CAT16",
    color_discrete_sequence=mycolors_discrete,
)
fig3.update_geos(fitbounds="locations", visible=False)
pyo.plot(fig3)

#%%

# buffered - regions with buffer for selected card

plottest = landusepolys[landusepolys["Card"] == testcard]
plottest.to_crs(epsg=4326, inplace=True)

fig1 = px.choropleth_mapbox(
    plottest,
    geojson=plottest.geometry,
    locations=plottest.index,
    center={"lat": -32, "lon": 116},
    zoom=10,
    mapbox_style="light",
    opacity=0.5,
    hover_name="Card",
    color="region_type",
    color_discrete_sequence=mycolors_discrete,
)
fig1.update_geos(fitbounds="locations", visible=False)
pyo.plot(fig1)

#%%

# original - convex hull polygon around region stops for the selected card

regionpolys = allregionpolys[allregionpolys["Card"] == testcard]

regionpolys.to_crs(epsg=4326, inplace=True)
regionpolys.reset_index(inplace=True)

regionpolys["regionid"] = (
    regionpolys["Card"].astype(str) + "-" + regionpolys["region_cluster"].astype(str)
)

fig2 = px.choropleth_mapbox(
    regionpolys,
    geojson=regionpolys.geometry,
    locations=regionpolys.index,
    center={"lat": -32, "lon": 116},
    zoom=10,
    mapbox_style="light",
    opacity=0.5,
    hover_name="Card",
    color="region_type",
    color_discrete_sequence=mycolors_discrete,
)
fig2.update_geos(fitbounds="locations", visible=False)
pyo.plot(fig2)
