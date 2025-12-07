-- TwinDB v0.1 Schema for PostgreSQL + TimescaleDB
-- Twin Data Model (TDM) implementation
-- Based on the VLDB/SIGMOD-level design from brainstorms.md

-- Asset: Physical objects (buildings, substations, etc.)
-- Key fields extracted from JSONB for high-frequency query dimensions
CREATE TABLE IF NOT EXISTS asset (
    aid           TEXT PRIMARY KEY,
    type          TEXT NOT NULL,          -- 'building', 'substation', ...
    city          TEXT,
    climate_zone  TEXT,
    area_m2       DOUBLE PRECISION,       -- heated floor area
    usage_type    TEXT,                   -- 'residential', 'office', 'school', ...
    year_built    INT,
    year_renovated INT,
    energy_label  TEXT,                   -- 'A','B','C','D','E','F'
    network_id    TEXT,                   -- district heating network / feeder id
    geo_location  TEXT,                   -- coordinates or postal code
    dataset_id    TEXT,                   -- source dataset: 'DanishHeat', 'BDG2'
    attrs         JSONB DEFAULT '{}'      -- additional attributes
);

-- Twin: Digital representation of an asset
CREATE TABLE IF NOT EXISTS twin (
    tid          TEXT PRIMARY KEY,
    aid          TEXT REFERENCES asset(aid),
    twin_type    TEXT,                    -- 'building', 'substation', 'district'
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    metadata     JSONB DEFAULT '{}'       -- resolution, meter_id, calibration info
);

-- Model: Simulation/prediction models (first-class citizen)
CREATE TABLE IF NOT EXISTS model (
    mid          TEXT PRIMARY KEY,
    scope        TEXT,                    -- 'building', 'substation', 'district'
    model_type   TEXT,                    -- 'phys', 'ml', 'hybrid'
    version      TEXT,
    location     TEXT,                    -- docker image, FMU path, HTTP endpoint
    io_schema    JSONB DEFAULT '{}'       -- input/output variable descriptions
);

-- Scenario: Configuration for what-if analysis (first-class citizen)
-- cfg_json schema: {weather_profile, price_profile, control_policy, retrofit_package}
CREATE TABLE IF NOT EXISTS scenario (
    cid          TEXT PRIMARY KEY,
    name         TEXT,
    cfg          JSONB NOT NULL,          -- scenario configuration
    base_cid     TEXT REFERENCES scenario(cid),  -- inheritance support
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- TimeSeries: Unified storage for historical (cid='REALITY') and simulated data
-- source field distinguishes: 'sensor' vs 'sim:model_id@version'
CREATE TABLE IF NOT EXISTS timeseries (
    tid          TEXT REFERENCES twin(tid),
    cid          TEXT NOT NULL,           -- 'REALITY' or FK -> scenario.cid
    metric       TEXT NOT NULL,           -- 'heat_load', 'temp_supply', ...
    metric_type  TEXT,                    -- 'power', 'energy', 'temp', 'flow', 'price'
    unit         TEXT,                    -- 'kW', 'kWh', 'degC', 'm3/h', 'DKK'
    ts           TIMESTAMPTZ NOT NULL,
    value        DOUBLE PRECISION,
    source       TEXT,                    -- 'sensor' | 'sim:mid@version'
    quality_flag TEXT,                    -- 'raw', 'interpolated', 'filled'
    PRIMARY KEY (tid, cid, metric, ts)
);

-- Simulation cache for scenario-aware optimization
-- Signature = hash(tid, cid, mid, window, metrics)
CREATE TABLE IF NOT EXISTS sim_cache (
    sig           TEXT PRIMARY KEY,
    tid           TEXT,
    cid           TEXT,
    mid           TEXT,
    window_start  TIMESTAMPTZ,
    window_end    TIMESTAMPTZ,
    metrics       TEXT[],
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common query patterns (as specified in brainstorm)
CREATE INDEX IF NOT EXISTS idx_asset_network ON asset(network_id);
CREATE INDEX IF NOT EXISTS idx_asset_usage ON asset(usage_type);
CREATE INDEX IF NOT EXISTS idx_asset_city ON asset(city);
CREATE INDEX IF NOT EXISTS idx_asset_year ON asset(year_built);
CREATE INDEX IF NOT EXISTS idx_twin_aid ON twin(aid);
CREATE INDEX IF NOT EXISTS idx_ts_tid_cid ON timeseries(tid, cid);
CREATE INDEX IF NOT EXISTS idx_ts_metric ON timeseries(metric);
CREATE INDEX IF NOT EXISTS idx_ts_ts ON timeseries(ts);

-- Insert default REALITY scenario for historical data
INSERT INTO scenario (cid, name, cfg) 
VALUES ('REALITY', 'Historical Reality', '{}')
ON CONFLICT (cid) DO NOTHING;
