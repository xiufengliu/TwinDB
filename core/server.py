"""
TwinDB REST API Server
FastAPI backend for the TwinDB web UI
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import os

from db import get_connection, execute_query, execute_many, execute_sql

app = FastAPI(title="TwinDB API", version="0.1.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models
# ============================================================================

class TimeSeriesPoint(BaseModel):
    ts: str
    value: float

class UploadTimeSeriesRequest(BaseModel):
    tid: str
    metric: str
    data: List[TimeSeriesPoint]

class Asset(BaseModel):
    aid: str
    type: str
    city: Optional[str] = None
    area_m2: Optional[float] = None
    usage_type: Optional[str] = None
    year_built: Optional[int] = None
    energy_label: Optional[str] = None
    network_id: Optional[str] = None
    attrs: Optional[Dict[str, Any]] = {}

class Twin(BaseModel):
    tid: str
    aid: str
    twin_type: str
    metadata: Optional[Dict[str, Any]] = {}

class Scenario(BaseModel):
    cid: str
    name: str
    cfg: Dict[str, Any] = {}

class QueryRequest(BaseModel):
    query: str

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# --- Assets ---
@app.get("/api/assets")
async def get_assets():
    """Get all assets"""
    try:
        df = execute_query("SELECT * FROM asset ORDER BY aid")
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/assets")
async def create_asset(asset: Asset):
    """Create a new asset"""
    try:
        execute_sql("""
            INSERT INTO asset (aid, type, city, area_m2, usage_type, year_built, energy_label, network_id, attrs)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (asset.aid, asset.type, asset.city, asset.area_m2, asset.usage_type, 
              asset.year_built, asset.energy_label, asset.network_id, 
              '{}' if not asset.attrs else str(asset.attrs).replace("'", '"')))
        return {"status": "created", "aid": asset.aid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Twins ---
@app.get("/api/twins")
async def get_twins():
    """Get all twins"""
    try:
        df = execute_query("SELECT * FROM twin ORDER BY tid")
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/twins")
async def create_twin(twin: Twin):
    """Create a new twin"""
    try:
        execute_sql("""
            INSERT INTO twin (tid, aid, twin_type, metadata)
            VALUES (%s, %s, %s, %s)
        """, (twin.tid, twin.aid, twin.twin_type, 
              '{}' if not twin.metadata else str(twin.metadata).replace("'", '"')))
        return {"status": "created", "tid": twin.tid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Scenarios ---
@app.get("/api/scenarios")
async def get_scenarios():
    """Get all scenarios"""
    try:
        df = execute_query("SELECT * FROM scenario ORDER BY cid")
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scenarios")
async def create_scenario(scenario: Scenario):
    """Create a new scenario"""
    try:
        execute_sql("""
            INSERT INTO scenario (cid, name, cfg)
            VALUES (%s, %s, %s)
            ON CONFLICT (cid) DO UPDATE SET name = EXCLUDED.name, cfg = EXCLUDED.cfg
        """, (scenario.cid, scenario.name, str(scenario.cfg).replace("'", '"')))
        return {"status": "created", "cid": scenario.cid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Models ---
@app.get("/api/models")
async def get_models():
    """Get all models"""
    try:
        df = execute_query("SELECT * FROM model ORDER BY mid")
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Time Series ---
@app.post("/api/timeseries/upload")
async def upload_timeseries(request: UploadTimeSeriesRequest):
    """Upload time series data for a twin"""
    try:
        # Prepare data for bulk insert
        data = [
            (request.tid, 'REALITY', request.metric, 'energy', 'kWh',
             point.ts, point.value, 'sensor', 'raw')
            for point in request.data
        ]
        
        execute_many("""
            INSERT INTO timeseries (tid, cid, metric, metric_type, unit, ts, value, source, quality_flag)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tid, cid, metric, ts) DO UPDATE SET value = EXCLUDED.value
        """, data)
        
        return {"status": "success", "count": len(data), "tid": request.tid, "metric": request.metric}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/timeseries/{tid}")
async def get_timeseries(tid: str, metric: Optional[str] = None, 
                         start: Optional[str] = None, end: Optional[str] = None):
    """Get time series data for a twin"""
    try:
        sql = "SELECT ts, metric, value FROM timeseries WHERE tid = %s AND cid = 'REALITY'"
        params = [tid]
        
        if metric:
            sql += " AND metric = %s"
            params.append(metric)
        if start:
            sql += " AND ts >= %s"
            params.append(start)
        if end:
            sql += " AND ts <= %s"
            params.append(end)
        
        sql += " ORDER BY ts"
        
        df = execute_query(sql, params)
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/timeseries/{tid}/metrics")
async def get_twin_metrics(tid: str):
    """Get available metrics for a twin"""
    try:
        df = execute_query("""
            SELECT metric, 
                   COUNT(*) as point_count,
                   MIN(ts) as start_ts,
                   MAX(ts) as end_ts
            FROM timeseries 
            WHERE tid = %s AND cid = 'REALITY'
            GROUP BY metric
            ORDER BY metric
        """, [tid])
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Query Execution (TwinQL) ---
@app.post("/api/query")
async def execute_twinql(request: QueryRequest):
    """Execute a TwinQL query"""
    try:
        # Import the TwinQL compiler/executor
        from compiler import compile_query
        from operators import execute_plan
        
        plan = compile_query(request.query)
        result = execute_plan(plan)
        
        return {
            "columns": list(result.columns),
            "data": result.values.tolist(),
            "execution_time_ms": 50,  # TODO: actual timing
            "explain": str(plan)
        }
    except ImportError:
        # Fallback if TwinQL not fully implemented
        raise HTTPException(status_code=501, detail="TwinQL execution not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("TWINDB_API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
