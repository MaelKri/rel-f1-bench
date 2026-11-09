"""Hand-crafted driver features with a strict per-row temporal cutoff (date < t on every join).

Windows are the last k known races, not calendar windows: the test db stops in Nov 2009
while test timestamps go up to 2016, so calendar windows would be empty at test time.
"""
import duckdb
import numpy as np
import pandas as pd

TABLES = ["results", "qualifying", "standings", "constructor_standings", "drivers"]


def _register(con, db, labels):
    for name in TABLES:
        con.register(name, db.table_dict[name].df)
    con.register("labels", labels)


SQL = """
WITH
-- driver race history before t, most recent first
hist AS (
    SELECT l.row_id, r.date, r.positionOrder AS pos, r.grid, r.points,
           CAST(r.position IS NULL AS INT) AS dnf,
           CAST(r.positionOrder <= 3 AS INT) AS podium,
           r.constructorId,
           row_number() OVER (PARTITION BY l.row_id ORDER BY r.date DESC, r.resultId DESC) AS rn
    FROM labels l JOIN results r ON r.driverId = l.driverId AND r.date < l.date
),
drv AS (
    SELECT row_id,
        count(*)                                  AS n_races,
        avg(pos)                                  AS pos_mean_all,
        avg(points)                               AS pts_mean_all,
        avg(dnf)                                  AS dnf_rate_all,
        avg(podium)                               AS podium_rate_all,
        {windows}
        arg_min(pos, rn)                          AS pos_last,
        arg_min(grid, rn)                         AS grid_last,
        arg_min(dnf, rn)                          AS dnf_last,
        arg_min(constructorId, rn)                AS constructor_last,
        max(date)                                 AS last_date,
        min(date)                                 AS first_date
    FROM hist GROUP BY row_id
),
-- qualifying history (table starts in 1994)
qhist AS (
    SELECT l.row_id, q.position AS qpos,
           row_number() OVER (PARTITION BY l.row_id ORDER BY q.date DESC, q.qualifyId DESC) AS rn
    FROM labels l JOIN qualifying q ON q.driverId = l.driverId AND q.date < l.date
),
qual AS (
    SELECT row_id,
        count(*)                                        AS n_quali,
        avg(qpos)                                       AS qpos_mean_all,
        avg(qpos) FILTER (WHERE rn <= 5)                AS qpos_mean_5,
        avg(CAST(qpos <= 3 AS INT)) FILTER (WHERE rn <= 5) AS qtop3_rate_5,
        arg_min(qpos, rn)                               AS qpos_last
    FROM qhist GROUP BY row_id
),
-- last known championship standing (cumulative within a season)
stand AS (
    SELECT l.row_id,
        arg_max(s.points, s.date)   AS champ_points,
        arg_max(s.position, s.date) AS champ_position,
        arg_max(s.wins, s.date)     AS champ_wins
    FROM labels l JOIN standings s ON s.driverId = l.driverId AND s.date < l.date
    GROUP BY l.row_id
),
-- 2-hop: strength of the driver's last known team
team_ctx AS (
    SELECT d.row_id, d.constructor_last, l.date
    FROM drv d JOIN labels l USING (row_id)
),
team_hist AS (
    SELECT t.row_id, r.positionOrder AS pos, r.points,
           dense_rank() OVER (PARTITION BY t.row_id ORDER BY r.date DESC) AS rk
    FROM team_ctx t JOIN results r ON r.constructorId = t.constructor_last AND r.date < t.date
),
team AS (
    SELECT row_id,
        avg(pos)    FILTER (WHERE rk <= 10) AS team_pos_mean_10,
        avg(points) FILTER (WHERE rk <= 10) AS team_pts_mean_10
    FROM team_hist GROUP BY row_id
),
team_stand AS (
    SELECT t.row_id,
        arg_max(cs.points, cs.date)   AS team_champ_points,
        arg_max(cs.position, cs.date) AS team_champ_position
    FROM team_ctx t JOIN constructor_standings cs
         ON cs.constructorId = t.constructor_last AND cs.date < t.date
    GROUP BY t.row_id
)
SELECT l.row_id,
    date_diff('day', dv.dob, l.date) / 365.25            AS age,
    date_diff('day', d.first_date, d.last_date) / 365.25 AS experience_years,
    date_diff('day', d.last_date, l.date)                AS days_since_last_race,
    d.* EXCLUDE (row_id, constructor_last, last_date, first_date),
    q.* EXCLUDE (row_id),
    s.* EXCLUDE (row_id),
    tm.* EXCLUDE (row_id),
    ts.* EXCLUDE (row_id)
FROM labels l
JOIN drivers dv ON dv.driverId = l.driverId
LEFT JOIN drv d         USING (row_id)
LEFT JOIN qual q        USING (row_id)
LEFT JOIN stand s       USING (row_id)
LEFT JOIN team tm       USING (row_id)
LEFT JOIN team_stand ts USING (row_id)
ORDER BY l.row_id
"""


def _window_sql(windows):
    lines = []
    for k in windows:
        lines += [
            f"avg(pos)    FILTER (WHERE rn <= {k}) AS pos_mean_{k},",
            f"avg(grid)   FILTER (WHERE rn <= {k}) AS grid_mean_{k},",
            f"avg(points) FILTER (WHERE rn <= {k}) AS pts_mean_{k},",
            f"avg(dnf)    FILTER (WHERE rn <= {k}) AS dnf_rate_{k},",
            f"avg(podium) FILTER (WHERE rn <= {k}) AS podium_rate_{k},",
        ]
    return "\n        ".join(lines)


def build_features(db, table, time_col, entity_col, windows=(5, 10), staleness=False):
    """Feature matrix aligned row by row with `table` (a RelBench Table or a DataFrame)."""
    df = table.df if hasattr(table, "df") else table
    labels = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "date": df[time_col].to_numpy(),
        "driverId": df[entity_col].to_numpy(),
    })
    con = duckdb.connect()
    _register(con, db, labels)
    X = con.execute(SQL.format(windows=_window_sql(windows))).df().drop(columns="row_id")
    con.close()
    for k in windows:  # recent form minus career form
        X[f"pos_trend_{k}"] = X[f"pos_mean_{k}"] - X["pos_mean_all"]
    if not staleness:
        X = X.drop(columns="days_since_last_race")
    return X.astype("float64")


def build_split_features(db, task, tables, **kw):
    """Features for the train, val and test tables of a task."""
    return {s: build_features(db, t, task.time_col, task.entity_col, **kw) for s, t in tables.items()}


# ---------------------------------------------------------------- leakage test
def _truncate(db, t):
    """Copy of the db keeping only rows strictly before t (static tables untouched)."""
    from relbench.base import Database, Table
    out = {}
    for name, tab in db.table_dict.items():
        df = tab.df if tab.time_col is None else tab.df[tab.df[tab.time_col] < t]
        out[name] = Table(df=df, fkey_col_to_pkey_table=tab.fkey_col_to_pkey_table,
                          pkey_col=tab.pkey_col, time_col=tab.time_col)
    return Database(out)


def leakage_test(db, table, time_col, entity_col, n=50, seed=0, **kw):
    """Features on the full db must equal features on the db truncated at each row's own time.

    Any difference means a feature read the future. Raises an AssertionError if so.
    """
    df = table.df if hasattr(table, "df") else table
    full = build_features(db, df, time_col, entity_col, **kw)
    rows = np.random.default_rng(seed).choice(len(df), size=min(n, len(df)), replace=False)
    for i in rows:
        row = df.iloc[[i]]
        trunc = build_features(_truncate(db, row[time_col].iloc[0]), row, time_col, entity_col, **kw)
        pd.testing.assert_frame_equal(full.iloc[[i]].reset_index(drop=True), trunc,
                                      check_exact=False, rtol=1e-9)
