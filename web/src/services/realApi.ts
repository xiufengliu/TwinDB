// Real API service for TwinDB backend
// Connects to the FastAPI server

import type {
    Asset, Twin, Scenario, Model, QueryResult,
    CompareScenarioRequest, UploadTimeSeriesRequest
} from '../types';

// API base URL - can be configured via environment
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============================================================================
// Helper Functions
// ============================================================================

async function fetchApi<T>(
    endpoint: string,
    options?: RequestInit
): Promise<T> {
    const url = `${API_BASE}${endpoint}`;
    const response = await fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options?.headers,
        },
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
}

// ============================================================================
// API Functions
// ============================================================================

export async function fetchTwins(): Promise<Twin[]> {
    return fetchApi<Twin[]>('/api/twins');
}

export async function fetchAssets(): Promise<Asset[]> {
    return fetchApi<Asset[]>('/api/assets');
}

export async function fetchScenarios(): Promise<Scenario[]> {
    return fetchApi<Scenario[]>('/api/scenarios');
}

export async function fetchModels(): Promise<Model[]> {
    return fetchApi<Model[]>('/api/models');
}

export async function createAsset(asset: Asset): Promise<Asset> {
    await fetchApi('/api/assets', {
        method: 'POST',
        body: JSON.stringify(asset),
    });
    return asset;
}

export async function createTwin(twin: Twin): Promise<Twin> {
    await fetchApi('/api/twins', {
        method: 'POST',
        body: JSON.stringify(twin),
    });
    return twin;
}

export async function updateAsset(aid: string, updates: Partial<Asset>): Promise<Asset> {
    // For now, re-create the asset with updates
    const updated = { aid, ...updates } as Asset;
    await fetchApi('/api/assets', {
        method: 'POST',
        body: JSON.stringify(updated),
    });
    return updated;
}

export async function updateTwin(tid: string, updates: Partial<Twin>): Promise<Twin> {
    const updated = { tid, ...updates } as Twin;
    await fetchApi('/api/twins', {
        method: 'POST',
        body: JSON.stringify(updated),
    });
    return updated;
}

export async function createScenario(scenario: Scenario): Promise<Scenario> {
    await fetchApi('/api/scenarios', {
        method: 'POST',
        body: JSON.stringify(scenario),
    });
    return scenario;
}

export async function executeQuery(query: string): Promise<QueryResult> {
    return fetchApi<QueryResult>('/api/query', {
        method: 'POST',
        body: JSON.stringify({ query }),
    });
}

export async function compareScenario(request: CompareScenarioRequest): Promise<QueryResult> {
    // Build a TwinQL query from the compare request
    const query = `
        SELECT period, baseline, retrofit, saving
        FROM COMPARE '${request.scenario_id}'
        FOR TWIN '${request.tid}'
        WINDOW '${request.window_start}' TO '${request.window_end}'
        METRIC '${request.metric}'
    `;
    return executeQuery(query);
}

export async function uploadTimeSeries(request: UploadTimeSeriesRequest): Promise<{ count: number }> {
    return fetchApi<{ count: number }>('/api/timeseries/upload', {
        method: 'POST',
        body: JSON.stringify(request),
    });
}

export async function getTimeSeries(
    tid: string,
    metric?: string,
    start?: string,
    end?: string
): Promise<{ ts: string; metric: string; value: number }[]> {
    const params = new URLSearchParams();
    if (metric) params.append('metric', metric);
    if (start) params.append('start', start);
    if (end) params.append('end', end);

    const queryString = params.toString() ? `?${params.toString()}` : '';
    return fetchApi(`/api/timeseries/${tid}${queryString}`);
}

export async function getTwinMetrics(tid: string): Promise<{
    metric: string;
    point_count: number;
    start_ts: string;
    end_ts: string;
}[]> {
    return fetchApi(`/api/timeseries/${tid}/metrics`);
}

// ============================================================================
// Export API object
// ============================================================================

export const realApi = {
    fetchTwins,
    fetchAssets,
    fetchScenarios,
    fetchModels,
    createAsset,
    createTwin,
    updateAsset,
    updateTwin,
    createScenario,
    executeQuery,
    compareScenario,
    uploadTimeSeries,
    getTimeSeries,
    getTwinMetrics,
};

export default realApi;
