// TwinDB Frontend Type Definitions
// Aligned with backend schema.sql

// ============================================================================
// Core Entity Types
// ============================================================================

export interface Asset {
    aid: string;
    type: 'building' | 'substation' | 'equipment' | 'district';
    city?: string;
    climate_zone?: string;
    area_m2?: number;
    usage_type?: string;
    year_built?: number;
    year_renovated?: number;
    energy_label?: string;
    network_id?: string;
    geo_location?: string;
    dataset_id?: string;
    attrs: Record<string, unknown>;
}

export interface Twin {
    tid: string;
    aid: string;
    twin_type: string;
    created_at?: string;
    metadata: Record<string, unknown>;
}

export interface Model {
    mid: string;
    scope: 'building' | 'substation' | 'district';
    model_type: 'phys' | 'ml' | 'hybrid';
    version: string;
    location: string;
    io_schema: Record<string, unknown>;
}

export interface Scenario {
    cid: string;
    name: string;
    cfg: ScenarioConfig;
    base_cid?: string;
    created_at?: string;
}

export interface ScenarioConfig {
    weather_profile?: string;
    price_profile?: string;
    control_policy?: {
        type: string;
        params: Record<string, number>;
    };
    retrofit_package?: RetrofitPackage;
}

export interface RetrofitPackage {
    id?: string;
    wall_u_before?: number;
    wall_u_after?: number;
    roof_u_before?: number;
    roof_u_after?: number;
    window_u_before?: number;
    window_u_after?: number;
    infiltration_before?: number;
    infiltration_after?: number;
    indoor_setpoint?: number;
}

export interface TimeSeriesPoint {
    ts: string;
    value: number;
    metric: string;
    source?: string;
    quality_flag?: string;
}

// ============================================================================
// Canvas Node Types (for React Flow)
// ============================================================================

export interface TwinNode {
    id: string;
    type: 'twin';
    position: { x: number; y: number };
    data: {
        twin: Twin;
        asset: Asset;
        hasData: boolean;
        metrics: string[];
        model?: Model;
    };
}

export interface Connection {
    id: string;
    source: string;
    target: string;
    type: 'network' | 'feeds' | 'parent';
    label?: string;
}

// ============================================================================
// Query Types
// ============================================================================

export interface QueryResult {
    columns: string[];
    data: unknown[][];
    explain: string;
    execution_time_ms: number;
}

export interface QueryError {
    message: string;
    line?: number;
    column?: number;
    suggestion?: string;
}

export interface SavedQuery {
    id: string;
    name: string;
    query: string;
    created_at: string;
    is_pinned: boolean;
}

// ============================================================================
// API Request/Response Types
// ============================================================================

export interface ExecuteQueryRequest {
    query: string;
}

export interface CompareScenarioRequest {
    tid: string;
    scenario_id: string;
    model_id: string;
    window_start: string;
    window_end: string;
    metric: string;
    agg_by?: 'hour' | 'day' | 'month' | 'year';
}

export interface UploadTimeSeriesRequest {
    tid: string;
    metric: string;
    data: Array<{ ts: string; value: number }>;
}

// ============================================================================
// State Types
// ============================================================================

export interface WorkspaceState {
    twins: Map<string, Twin>;
    assets: Map<string, Asset>;
    scenarios: Map<string, Scenario>;
    models: Map<string, Model>;
    selectedTwinId: string | null;
    zoom: number;
    pan: { x: number; y: number };
}

export interface QueryState {
    currentQuery: string;
    isExecuting: boolean;
    results: QueryResult | null;
    error: QueryError | null;
    history: SavedQuery[];
}

export type ViewMode = 'workspace' | 'console' | 'split';
