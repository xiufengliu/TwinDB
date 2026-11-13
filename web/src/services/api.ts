// API service for TwinDB backend
// Toggle USE_REAL_API to switch between mock (demo) and real (FastAPI) backend

import type {
    Asset, Twin, Scenario, Model, QueryResult,
    CompareScenarioRequest, UploadTimeSeriesRequest
} from '../types';
import { realApi } from './realApi';

// ============================================================================
// Configuration - Set to true when FastAPI backend is running
// ============================================================================
const USE_REAL_API = import.meta.env.VITE_USE_REAL_API === 'true' || false;

// ============================================================================
// Mock Data
// ============================================================================

const mockTwins: Twin[] = [
    { tid: 'Twin_B101', aid: 'A101', twin_type: 'building', metadata: { resolution: 'hourly' } },
    { tid: 'Twin_B102', aid: 'A102', twin_type: 'building', metadata: { resolution: 'hourly' } },
    { tid: 'Twin_B123', aid: 'A123', twin_type: 'building', metadata: { resolution: 'hourly' } },
    { tid: 'Twin_SUB01', aid: 'SUB01', twin_type: 'substation', metadata: {} },
];

const mockAssets: Asset[] = [
    { aid: 'A101', type: 'building', city: 'Copenhagen', area_m2: 2100, usage_type: 'residential', year_built: 1975, energy_label: 'D', network_id: 'DH_CPH_01', attrs: {} },
    { aid: 'A102', type: 'building', city: 'Copenhagen', area_m2: 3200, usage_type: 'office', year_built: 1990, energy_label: 'C', network_id: 'DH_CPH_01', attrs: {} },
    { aid: 'A123', type: 'building', city: 'Copenhagen', area_m2: 2500, usage_type: 'residential', year_built: 1985, energy_label: 'D', network_id: 'DH_CPH_01', attrs: {} },
    { aid: 'SUB01', type: 'substation', city: 'Copenhagen', network_id: 'DH_CPH_01', attrs: { capacity_kw: 5000 } },
];

const mockScenarios: Scenario[] = [
    { cid: 'REALITY', name: 'Historical Reality', cfg: {}, created_at: '2024-01-01' },
    { cid: 'deep_retrofit', name: 'Deep Retrofit', cfg: { retrofit_package: { wall_u_after: 0.18, window_u_after: 0.90 } } },
    { cid: 's_retrofit', name: 'Medium Retrofit', cfg: { retrofit_package: { wall_u_after: 0.25, window_u_after: 1.2 } } },
];

const mockModels: Model[] = [
    { mid: 'BuildingHeat_v3', scope: 'building', model_type: 'phys', version: '3.0', location: 'docker://twindb/building-heat:v3', io_schema: {} },
    { mid: 'SubstationFlow_v1', scope: 'substation', model_type: 'hybrid', version: '1.0', location: 'docker://twindb/substation-flow:v1', io_schema: {} },
];

// ============================================================================
// API Functions
// ============================================================================

export async function fetchTwins(): Promise<Twin[]> {
    await delay(300);
    return mockTwins;
}

export async function fetchAssets(): Promise<Asset[]> {
    await delay(300);
    return mockAssets;
}

export async function fetchScenarios(): Promise<Scenario[]> {
    await delay(200);
    return mockScenarios;
}

export async function fetchModels(): Promise<Model[]> {
    await delay(200);
    return mockModels;
}

export async function createAsset(asset: Asset): Promise<Asset> {
    await delay(300);
    mockAssets.push(asset);
    return asset;
}

export async function createTwin(twin: Twin): Promise<Twin> {
    await delay(300);
    mockTwins.push(twin);
    return twin;
}

export async function updateAsset(aid: string, updates: Partial<Asset>): Promise<Asset> {
    await delay(300);
    const idx = mockAssets.findIndex(a => a.aid === aid);
    if (idx >= 0) {
        mockAssets[idx] = { ...mockAssets[idx], ...updates };
        return mockAssets[idx];
    }
    throw new Error(`Asset ${aid} not found`);
}

export async function updateTwin(tid: string, updates: Partial<Twin>): Promise<Twin> {
    await delay(300);
    const idx = mockTwins.findIndex(t => t.tid === tid);
    if (idx >= 0) {
        mockTwins[idx] = { ...mockTwins[idx], ...updates };
        return mockTwins[idx];
    }
    throw new Error(`Twin ${tid} not found`);
}

export async function createScenario(scenario: Scenario): Promise<Scenario> {
    await delay(300);
    mockScenarios.push(scenario);
    return scenario;
}

export async function executeQuery(query: string): Promise<QueryResult> {
    await delay(800);

    // Parse query to simulate results
    if (query.toLowerCase().includes('error')) {
        throw new Error('Syntax error at line 3: unexpected token');
    }

    // Generate mock results
    const months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
        '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12'];

    const data = months.map(month => {
        const baseline = Math.round(20000 + Math.random() * 15000 + (month < '2018-04' || month > '2018-09' ? 10000 : 0));
        const retrofit = Math.round(baseline * (0.55 + Math.random() * 0.1));
        const saving = baseline - retrofit;
        const savingPct = Math.round((saving / baseline) * 100);
        return [month, baseline, retrofit, saving, `${savingPct}%`];
    });

    return {
        columns: ['period', 'baseline', 'retrofit', 'saving', 'saving_pct'],
        data,
        explain: `Join(on=period) [card=12]
  Agg(by=month, funcs={value: sum}) [card=12]
    Hist(tid=Twin_B123, window=(2018-01-01, 2019-01-01)) [card=8760]
  Agg(by=month, funcs={value: sum}) [card=12]
    Sim(tid=Twin_B123, scenario=deep_retrofit, model=BuildingHeat_v3) [card=8760]`,
        execution_time_ms: 54,
    };
}

export async function compareScenario(request: CompareScenarioRequest): Promise<QueryResult> {
    await delay(600);

    const months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
        '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12'];

    const data = months.map(month => {
        const baseline = Math.round(20000 + Math.random() * 15000 + (month < '2018-04' || month > '2018-09' ? 10000 : 0));
        const retrofit = Math.round(baseline * (0.55 + Math.random() * 0.1));
        return [month, baseline, retrofit, baseline - retrofit];
    });

    return {
        columns: ['period', 'baseline', 'retrofit', 'saving'],
        data,
        explain: `Scenario comparison for ${request.tid}`,
        execution_time_ms: 54,
    };
}

export async function uploadTimeSeries(request: UploadTimeSeriesRequest): Promise<{ count: number }> {
    await delay(500);
    return { count: request.data.length };
}

// ============================================================================
// Helpers
// ============================================================================

function delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Mock API object
const mockApi = {
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
};

// Export either real or mock API based on configuration
export const api = USE_REAL_API ? realApi : mockApi;

export default api;
