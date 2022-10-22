from shapely.geometry import Point, MultiPoint

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN
import datetime as dt

#%%

def create_point(row):
    """Returns a shapely point object based on values in x and y columns"""
    point = Point(row["X"], row["Y"])
    return point

def create_geo_stops(trainStops, busStops):
    trainStops.rename(
        columns={"StationRefNo": "StopID", "Station": "StopName"}, inplace=True
    )
    busStops.rename(
        columns={"BusStopId": "StopID", "BusStopName": "StopName"}, inplace=True
    )

    stops = pd.concat([busStops, trainStops])
    stopTable = stops[["StopID", "X", "Y"]]
    stopTable.drop_duplicates(subset="StopID", inplace=True)

    # need to split - some are in different CRSs. Where X > 130 assume 7844
    # (? something UTM) and otherwise 4326s
    # just drop for now
    stopTable = stopTable[stopTable["X"] < 130]

    # convert coords to Shapely geometry
    stopTable["geometry"] = stopTable.apply(create_point, axis=1)

    # convert to geodataframe
    geo = gpd.GeoDataFrame(stopTable, geometry="geometry")
    geo = geo.set_crs("epsg:4326", allow_override=True)
    return geo

#%%

def create_centroid(clusterID, geo, outcol):
    """Returns centroid of the cluster for each stop"""
    multi_point = MultiPoint(geo["geometry"][geo[outcol] == clusterID].to_list())
    return multi_point.centroid

def create_stop_cluster(geo, outcol, eps, minpts):
    geo = geo.to_crs(epsg=32749)
    geo["X"] = geo["geometry"].x
    geo["Y"] = geo["geometry"].y

    db = DBSCAN(eps=eps, min_samples=minpts, metric="euclidean", algorithm="ball_tree")
    geo[outcol] = db.fit_predict(geo[["Y", "X"]])

    geo.to_crs(epsg=4326, inplace=True)
    geo[outcol] = geo[outcol].astype(int)
    clusters = pd.DataFrame(
        data=np.arange(0, geo[outcol].max() + 1), columns=["clusterID"]
    )

    clusters["geometry"] = clusters.apply(
        lambda x: create_centroid(x["clusterID"], geo, outcol), axis=1
    )
    clusters = gpd.GeoDataFrame(clusters, geometry="geometry")
    clusters = clusters.set_crs(geo.crs, allow_override=True)

    return clusters, geo

#%%

def journey_processing(journeys, MIN_USES):
    TOTAL_LEN = len(
        journeys
    )
    # total length of original journeys dataset - to work out how much is being dropped

    print(TOTAL_LEN, "trips in original dataset")  # 9,643,083
    v = journeys["Cardid"].value_counts()
    print(len(v), " cards in data set")
    print(len(v.index[v.gt(MIN_USES)]), " cards kept")
    print(len(v) - len(v.index[v.gt(MIN_USES)]), " cards dropped")

    journeys = journeys[journeys["Cardid"].isin(v.index[v.gt(MIN_USES)])]

    DROPPED_MIN = TOTAL_LEN - len(
        journeys
    )  # how many records got dropped by not being above minimum number of uses
    print(
        DROPPED_MIN, "trips dropped due to card not being above minimum use threshold"
    )  # 1,741,315

    print(
        DROPPED_MIN / TOTAL_LEN * 100,
        "percent dropped due to card not being above minimum use threshold",
    )  # 18.06%

    # Replace unknown stops with a zero
    journeys["OnLocation"] = np.where(
        journeys["OnLocation"] == "Unknown", 0, journeys["OnLocation"]
    )
    journeys["OffLocation"] = np.where(
        journeys["OffLocation"] == "Unknown", 0, journeys["OffLocation"]
    )

    # Make sure all stops are numeric (to find duplicates)
    journeys["OnLocation"] = pd.to_numeric(journeys["OnLocation"])
    journeys["OffLocation"] = pd.to_numeric(journeys["OffLocation"])

    # Convert Date format to timestamp
    journeys["OnTime"] = pd.to_datetime(journeys["OnDate"], format="%Y%m%d%H%M%S")
    journeys["OffTime"] = pd.to_datetime(journeys["OffDate"], format="%Y%m%d%H%M%S")

    # Separate Hour and Date
    journeys["OnHour"] = pd.DatetimeIndex(journeys["OnTime"]).hour
    journeys["OffHour"] = pd.DatetimeIndex(journeys["OffTime"]).hour
    journeys["OnDate"] = pd.DatetimeIndex(journeys["OnTime"]).date
    journeys["OffDate"] = pd.DatetimeIndex(journeys["OffTime"]).date
    journeys["OnDay"] = pd.DatetimeIndex(journeys["OnTime"]).day

    # Calculate duration of trip, and time since last trip
    journeys["tripTime"] = journeys["OffTime"] - journeys["OnTime"]
    journeys["tripTime_h"] = journeys["tripTime"] / dt.timedelta(hours=1)

    journeys["timeSince"] = (journeys["OnTime"] - journeys["OffTime"].shift(1)).where(
        journeys["Cardid"] == journeys["Cardid"].shift(1)
    )
    # note this sometimes returns 0 for where there are synthetic activities

    # used for stays
    journeys["arriveHour"] = (
        journeys["OffHour"]
        .shift(1)
        .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
    )
    journeys["fromLocation"] = (
        journeys["OffLocation"]
        .shift(1)
        .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
    )
    journeys["originLocation"] = (
        journeys["OnLocation"]
        .shift(1)
        .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
    )
    # NEW - where did this stay originate?
    journeys["ArriveMode"] = (
        journeys["OnMode"]
        .shift(1)
        .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
    )
    # NEW - what mode was the prior trip

    journeys["OffTime_prev"] = (
        journeys["OffTime"]
        .shift(1)
        .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
    )
    journeys["OnTime_prev"] = (
        journeys["OnTime"]
        .shift(1)
        .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
    )

    journeys["tripTime_prev"] = journeys["OffTime_prev"] - journeys["OnTime_prev"]
    journeys["tripTime_h_prev"] = journeys["tripTime_prev"] / dt.timedelta(hours=1)

    # SYNTHETIC RECORDS
    # synthetic records are where the system backfills a tag on or off if someone
    # has forgotten to do so
    # this will result in inaccurate stay times/locations, as really we don't know
    # where someone has been between times they've actually used their cards

    journeys["SyntheticFlag"] = np.where(
        journeys["OnTran"].str.contains("synthetic", na=False, case=False)
        | journeys["OffTran"].str.contains("synthetic", na=False, case=False),
        1,
        0,
    )

    print(
        "Synthetic activities comprise ",
        journeys["SyntheticFlag"].sum() / len(journeys) * 100,
        "percent, or",
        journeys["SyntheticFlag"].sum(),
        "individual activities",
    )
    # 2.6% of current journeys data
    # 203,273 records

    # ACCEPTABLE SYNTHETIC ACTIVITIES
    # Warwick (stopID 2800) and Whitfords (stopID 2763)
    # "Within the interchange area, you may transfer from the train to the bus,
    # without tagging off the train or from the bus to the train, without tagging
    # onto the train. The SmartRider will automatically transfer you to/from the
    # train service. You will always need to tag on and off the bus."

    # In this logic:
    # OnTran
    # where this is synthetic and the previous tag off is a bus within x minutes (?)
    # OffTran
    # where this is synthetic and the next tag on is a bus within x minutes (?)

    # have used 60 minutes as this is consistent with the transfer logic

    journeys["SyntheticOnOk_WW"] = np.where(
        (journeys["OnTran"].str.contains("synthetic", na=False, case=False)) &
        # synthetic tag on
        ((journeys["OnLocation"] == 2800) | (journeys["OnLocation"] == 2763)) &
        # at Warwick or Whitfords
        (journeys["OnMode"].shift(1) == "Bus") &
        # last activity was a bus
        (
            (journeys["OffTime"] - journeys["OffTime"].shift(1))
            < pd.Timedelta(60, unit="m")
        )
        &
        # time between actual tag off and last tag off is <60min
        (journeys["Cardid"] == journeys["Cardid"].shift(1)),
        # last activity was with the same card
        1,
        0,
    )

    journeys["SyntheticOffOk_WW"] = np.where(
        (journeys["OffTran"].str.contains("synthetic", na=False, case=False)) &
        # synthetic tag off
        ((journeys["OffLocation"] == 2800) | (journeys["OffLocation"] == 2763)) &
        # at Warwick or Whitfords
        (journeys["OnMode"].shift(-1) == "Bus") &
        # next activity is a bus
        (
            (journeys["OnTime"].shift(-1) - journeys["OnTime"])
            < pd.Timedelta(60, unit="m")
        )
        &
        # time between actual next tag on and this tag on is <60min
        (journeys["Cardid"] == journeys["Cardid"].shift(-1)),
        # next activity is with the same card
        1,
        0,
    )

    # Bus transfers
    # The exception is when you tag onto two bus services within 60 minutes
    # without tagging off the first. When this happens, we use your second tag
    # on location as your tag off location for the first bus service.

    # OffTran
    # where this is synthetic and the next activity is a bus within 60 minutes

    # but if it's not very long, transferring between a train and a bus, or from
    # a train to a train, should be findable (and ok) too

    journeys["SyntheticOffOk_transfer"] = np.where(
        (journeys["OffTran"].str.contains("synthetic", na=False, case=False))
        & (
            (journeys["OnTime"].shift(-1) - journeys["OnTime"])
            < pd.Timedelta(60, unit="m")
        )
        & (journeys["Cardid"] == journeys["Cardid"].shift(1)),
        1,
        0,
    )

    print(
        "On ok at Whitfords/Warwick:",
        journeys["SyntheticOnOk_WW"].sum() / journeys["SyntheticFlag"].sum() * 100,
        "percent of synthetic activities, or",
        journeys["SyntheticOnOk_WW"].sum(),
        "individual activities",
    )

    print(
        "Off ok at Whitfords/Warwick:",
        journeys["SyntheticOffOk_WW"].sum() / journeys["SyntheticFlag"].sum() * 100,
        "percent of synthetic activities, or",
        journeys["SyntheticOffOk_WW"].sum(),
        "individual activities",
    )

    print(
        "Off ok transfers:",
        journeys["SyntheticOffOk_transfer"].sum()
        / journeys["SyntheticFlag"].sum()
        * 100,
        "percent of synthetic activities, or",
        journeys["SyntheticOffOk_transfer"].sum(),
        "individual activities",
    )

    # there can be double counting here - some of the off ok bus transfers
    # will be caught by the Warwick/Whitfords logic too

    # overwrite synthetic flags where they're ok
    journeys["SyntheticOk"] = (
        journeys["SyntheticOnOk_WW"]
        + journeys["SyntheticOffOk_WW"]
        + journeys["SyntheticOffOk_transfer"]
    )
    journeys["SyntheticFlag"] = np.where(
        journeys["SyntheticOk"] > 0, 0, journeys["SyntheticFlag"]
    )

    syn_pivot = pd.pivot_table(
        journeys, index="Cardid", columns="OnDay", values="SyntheticFlag", aggfunc="sum"
    )
    syn_sum = syn_pivot.melt(ignore_index=False)
    syn_sum.reset_index(inplace=True)

    journeys = journeys.merge(syn_sum, how="left", on=["OnDay", "Cardid"])
    journeys.rename(columns={"value": "SyntheticDrop"}, inplace=True)

    # so how many records would be dropped if we drop whole
    # days that include synthetic records?
    print(len(journeys[journeys["SyntheticDrop"] > 0]), "trips would be dropped")
    print(
        "... which is",
        len(journeys[journeys["SyntheticDrop"] > 0]) / len(journeys) * 100,
        "percent of the trips",
    )

    # drop synthetic activities that aren't salvageable
    journeys = journeys.drop(journeys[journeys["SyntheticDrop"] > 0].index).reset_index(
        drop=True
    )

    return journeys

#%%

def learn_stay_class(
    df, col_duration, col_arrival, col_token, col_distance, RETURN_THRESHOLD
):

    df[col_token] = np.where(
        df[col_token] == "Student 50 cent", "School", df[col_token]
    )
    df[col_token] = np.where(
        df[col_token] == "Student (Up to Yr 12)", "School", df[col_token]
    )
    df[col_token] = np.where(
        df[col_token] == "Student Tertiary", "Tertiary", df[col_token]
    )

    df[col_token] = np.where(df[col_token] == "Senior", "Concession", df[col_token])
    df[col_token] = np.where(
        df[col_token] == "Senior Off-Peak", "Concession", df[col_token]
    )
    df[col_token] = np.where(
        df[col_token] == "Health Care", "Concession", df[col_token]
    )
    df[col_token] = np.where(
        df[col_token] == "Pensioner Off-Peak", "Concession", df[col_token]
    )
    df[col_token] = np.where(df[col_token] == "Pensioner", "Concession", df[col_token])
    df[col_token] = np.where(
        df[col_token] == "PTA Free Pass", "Concession", df[col_token]
    )  # wouldn't that be employees?
    df[col_token] = np.where(df[col_token] == "Freerider", "Concession", df[col_token])
    df[col_token] = np.where(
        df[col_token] == "PTA Concession", "Concession", df[col_token]
    )
    df[col_token] = np.where(df[col_token] == "Veteran", "Concession", df[col_token])

    df["class"] = 0

    # A attractor activities

    df["class"] = np.where(
        (df[col_duration] >= 5)
        & (df[col_duration] < 16)
        & (df[col_arrival] > 5)
        & (df[col_arrival] <= 12)
        & (df[col_token] != "School"),
        1,
        df["class"],
    )  # work W

    df["class"] = np.where(
        (df[col_duration] >= 5)
        & (df[col_duration] < 16)
        & (df[col_arrival] > 5)
        & (df[col_arrival] <= 12)
        & (df[col_token] == "School"),
        2,
        df["class"],
    )  # school E

    df["class"] = np.where(
        (df[col_duration] >= 1) & (df[col_duration] < 3), 3, df["class"]
    )  # short stay S

    df["class"] = np.where(
        (df[col_duration] >= 3) & (df[col_duration] < 6), 4, df["class"]
    )  # medium stay M

    df["class"] = np.where(
        (df[col_arrival] > 12)
        & (df[col_arrival] < 18)
        & (df[col_duration] >= 6)
        & (df[col_duration] < 10),
        4,
        df["class"],
    )  # medium stay M

    # C Connector activities

    df["class"] = np.where(
        (df[col_duration] >= 0) & (df[col_duration] < 1), 8, df["class"]
    )  # overnight transfer (default) before 5am or after 7pm T

    df["class"] = np.where(
        (df[col_arrival] > 5)
        & (df[col_arrival] <= 9)
        & (df[col_duration] >= 0)
        & (df[col_duration] < 1),
        5,
        df["class"],
    )  # AM peak T

    df["class"] = np.where(
        (df[col_arrival] > 9)
        & (df[col_arrival] <= 15)
        & (df[col_duration] >= 0)
        & (df[col_duration] < 1),
        6,
        df["class"],
    )  # day time T

    df["class"] = np.where(
        (df[col_arrival] > 15)
        & (df[col_arrival] <= 19)
        & (df[col_duration] >= 0)
        & (df[col_duration] < 1),
        7,
        df["class"],
    )  # PM peak T

    # NEW
    # df['class'] = np.where((df[col_duration]>=0) & (df[col_duration]<1) &
    #                                 (df[col_distance] <= RETURN_THRESHOLD),
    #                                 10, df['class']) # short out and backs R

    # generator activities (overnight stays)

    df["class"] = np.where(
        (df[col_arrival] >= 12) & (df[col_duration] >= 10) & (df[col_duration] <= 20),
        9,
        df["class"],
    )  # sleep L
    df["class"] = np.where(
        (df[col_arrival] >= 18) & (df[col_duration] >= 6) & (df[col_duration] <= 20),
        9,
        df["class"],
    )  # sleep L
    df["class"] = np.where(
        (df[col_arrival] <= 5) & (df[col_duration] >= 6) & (df[col_duration] <= 20),
        9,
        df["class"],
    )  # sleep - early hours arrive L
    df["class"] = np.where(
        (df[col_arrival] <= 12) & (df[col_duration] >= 16) & (df[col_duration] <= 20),
        9,
        df["class"],
    )  # sleep - after AM arrive L
    df["class"] = np.where(
        (df[col_duration] > 20) & (df[col_duration] <= 3 * 24), 10, df["class"]
    )  # long stay >20h and <3 days V

    return df[col_token], df["class"]

#%%

def generate_regions(
    card,
    clusters,
    activities,
    STAY_DROP_THRESHOLD,
    ANCHOR_REGION_SIZE,
    MIN_POINTS_REGION,
    MAPBUFFER,
):
    carddata = activities[activities["Cardid"] == card]

    # mean, standard deviation, count, sum
    hist = carddata.groupby(["spatial_cluster"], as_index=False).agg(
        {"Duration": ["mean", "std", "count", "sum"]}
    )

    # remove multiindex
    hist.columns = [" ".join(col).strip() for col in hist.columns.values]

    # drop where number of records is below the threshold, otherwise
    # distribution is meaningless
    hist = hist.drop(
        hist[hist["Duration count"] < STAY_DROP_THRESHOLD].index
    ).reset_index(drop=True)

    # merge hist with clusters to get geometry for cluster ID
    hist = pd.merge(
        hist, clusters, left_on="spatial_cluster", right_on="clusterID", how="left"
    )
    hist.drop(["clusterID"], axis=1, inplace=True)

    hist = gpd.GeoDataFrame(hist, geometry="geometry")

    regions, hist = create_stop_cluster(
        hist, "region_cluster", ANCHOR_REGION_SIZE, MIN_POINTS_REGION
    )

    region_summary = pd.pivot_table(
        hist,
        index="region_cluster",
        values=("Duration count", "Duration sum"),
        aggfunc=("sum"),
    )
    region_summary.reset_index(inplace=True)

    region_summary.rename(
        columns={"Duration count": "num_visits", "Duration sum": "total_stay_time"},
        inplace=True,
    )

    region_summary["avg_stay_time"] = (
        region_summary["total_stay_time"] / region_summary["num_visits"]
    )

    region_summary["fraction_time"] = (
        region_summary["total_stay_time"] / region_summary["total_stay_time"].sum()
    )
    region_summary["fraction_time_cum"] = region_summary["fraction_time"].cumsum(axis=0)

    region_summary["fraction_visits"] = (
        region_summary["num_visits"] / region_summary["num_visits"].sum()
    )
    region_summary["fraction_visits_cum"] = region_summary["fraction_visits"].cumsum(
        axis=0
    )

    region_summary = pd.merge(
        region_summary,
        regions,
        left_on="region_cluster",
        right_on="clusterID",
        how="left",
    )
    region_summary.drop(["clusterID"], axis=1, inplace=True)

    region_summary = gpd.GeoDataFrame(region_summary, geometry="geometry")

    regionpolys = hist.dissolve("region_cluster").convex_hull
    regionpolys = gpd.GeoDataFrame(regionpolys)
    regionpolys.reset_index(inplace=True)
    regionpolys.rename(columns={0: "geometry"}, inplace=True)

    # buffer the region centroid to get the circle
    regionpolys = regionpolys.to_crs(epsg=32749)
    regionpolys["geometry"] = regionpolys["geometry"].buffer(MAPBUFFER)
    regionpolys["area"] = regionpolys.area
    regionpolys["area"] = regionpolys["area"] / 1000000  # to get sq km

    return hist, region_summary, regionpolys
