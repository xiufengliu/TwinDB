// Global state management using Zustand
import { create } from 'zustand';
import type { Twin, Asset, Scenario, Model, QueryResult, QueryError, SavedQuery, ViewMode } from '../types';

// ============================================================================
// Workspace Store - Model Builder State
// ============================================================================

// Connection/Relationship types for district heating networks
export type ConnectionType =
    | 'supplies_heat_to'    // Heat Plant → Pipeline → Building
    | 'connects_to'         // Pipeline → Substation, Equipment connections  
    | 'contains'            // District → Buildings
    | 'meters'              // Meter → Building/Pipeline
    | 'controlled_by'       // Equipment → Pump Station
    | 'feeds_into';         // Storage → Pipeline

export interface TwinConnection {
    id: string;
    source: string;         // Source twin ID
    target: string;         // Target twin ID
    type: ConnectionType;
    label?: string;
    metadata?: Record<string, unknown>;
}

interface WorkspaceStore {
    // Data
    twins: Map<string, Twin>;
    assets: Map<string, Asset>;
    scenarios: Map<string, Scenario>;
    models: Map<string, Model>;
    connections: TwinConnection[];

    // UI State
    selectedTwinId: string | null;
    zoom: number;
    pan: { x: number; y: number };

    // Actions
    setTwins: (twins: Twin[]) => void;
    setAssets: (assets: Asset[]) => void;
    setScenarios: (scenarios: Scenario[]) => void;
    setModels: (models: Model[]) => void;
    selectTwin: (tid: string | null) => void;
    addTwin: (twin: Twin, asset: Asset) => void;
    addScenario: (scenario: Scenario) => void;
    updateTwin: (tid: string, updates: Partial<Twin>) => void;
    updateAsset: (aid: string, updates: Partial<Asset>) => void;
    removeTwin: (tid: string) => void;
    setZoom: (zoom: number) => void;
    setPan: (pan: { x: number; y: number }) => void;
    // Connection actions
    addConnection: (connection: TwinConnection) => void;
    removeConnection: (id: string) => void;
    setConnections: (connections: TwinConnection[]) => void;
}

export const useWorkspaceStore = create<WorkspaceStore>((set) => ({
    twins: new Map(),
    assets: new Map(),
    scenarios: new Map(),
    models: new Map(),
    connections: [],
    selectedTwinId: null,
    zoom: 1,
    pan: { x: 0, y: 0 },

    setTwins: (twins) => set({ twins: new Map(twins.map(t => [t.tid, t])) }),
    setAssets: (assets) => set({ assets: new Map(assets.map(a => [a.aid, a])) }),
    setScenarios: (scenarios) => set({ scenarios: new Map(scenarios.map(s => [s.cid, s])) }),
    setModels: (models) => set({ models: new Map(models.map(m => [m.mid, m])) }),

    selectTwin: (tid) => set({ selectedTwinId: tid }),

    addTwin: (twin, asset) => set((state) => ({
        twins: new Map(state.twins).set(twin.tid, twin),
        assets: new Map(state.assets).set(asset.aid, asset),
    })),

    addScenario: (scenario) => set((state) => ({
        scenarios: new Map(state.scenarios).set(scenario.cid, scenario),
    })),

    updateTwin: (tid, updates) => set((state) => {
        const twin = state.twins.get(tid);
        if (!twin) return state;
        const newTwins = new Map(state.twins);
        newTwins.set(tid, { ...twin, ...updates });
        return { twins: newTwins };
    }),

    updateAsset: (aid, updates) => set((state) => {
        const asset = state.assets.get(aid);
        if (!asset) return state;
        const newAssets = new Map(state.assets);
        newAssets.set(aid, { ...asset, ...updates });
        return { assets: newAssets };
    }),

    removeTwin: (tid) => set((state) => {
        const twin = state.twins.get(tid);
        const newTwins = new Map(state.twins);
        const newAssets = new Map(state.assets);
        newTwins.delete(tid);
        if (twin) newAssets.delete(twin.aid);
        // Also remove any connections involving this twin
        const newConnections = state.connections.filter(
            c => c.source !== tid && c.target !== tid
        );
        return {
            twins: newTwins,
            assets: newAssets,
            connections: newConnections,
            selectedTwinId: state.selectedTwinId === tid ? null : state.selectedTwinId
        };
    }),

    setZoom: (zoom) => set({ zoom }),
    setPan: (pan) => set({ pan }),

    // Connection actions
    addConnection: (connection) => set((state) => ({
        connections: [...state.connections, connection]
    })),

    removeConnection: (id) => set((state) => ({
        connections: state.connections.filter(c => c.id !== id)
    })),

    setConnections: (connections) => set({ connections }),
}));

// ============================================================================
// Query Store - TwinQL Console State
// ============================================================================

interface QueryStore {
    currentQuery: string;
    isExecuting: boolean;
    results: QueryResult | null;
    error: QueryError | null;
    history: SavedQuery[];
    activeTab: 'table' | 'chart' | 'json';

    setQuery: (query: string) => void;
    setExecuting: (executing: boolean) => void;
    setResults: (results: QueryResult | null) => void;
    setError: (error: QueryError | null) => void;
    addToHistory: (query: SavedQuery) => void;
    setActiveTab: (tab: 'table' | 'chart' | 'json') => void;
    clearResults: () => void;
}

export const useQueryStore = create<QueryStore>((set) => ({
    currentQuery: `-- TwinQL Query Example
TWIN SELECT 
  h.value AS baseline,
  s.value AS retrofit
FROM 
  HISTORICAL twin_id='Twin_B123' 
    WINDOW '2018-01-01' TO '2019-01-01' 
    METRIC 'heat_load' 
    AGGREGATE BY month AS h,
  SIMULATE twin_id='Twin_B123' 
    SCENARIO deep_retrofit 
    MODEL 'BuildingHeat_v3'
    WINDOW '2018-01-01' TO '2019-01-01' 
    METRIC 'heat_load' 
    AGGREGATE BY month AS s`,
    isExecuting: false,
    results: null,
    error: null,
    history: [],
    activeTab: 'table',

    setQuery: (query) => set({ currentQuery: query }),
    setExecuting: (executing) => set({ isExecuting: executing }),
    setResults: (results) => set({ results, error: null }),
    setError: (error) => set({ error, results: null }),
    addToHistory: (query) => set((state) => ({
        history: [query, ...state.history].slice(0, 50)
    })),
    setActiveTab: (tab) => set({ activeTab: tab }),
    clearResults: () => set({ results: null, error: null }),
}));

// ============================================================================
// UI Store - Global UI State
// ============================================================================

interface UIStore {
    viewMode: ViewMode;
    sidebarOpen: boolean;
    propertiesPanelOpen: boolean;
    theme: 'dark' | 'light';
    notifications: Array<{ id: string; type: 'info' | 'success' | 'warning' | 'error'; message: string }>;

    setViewMode: (mode: ViewMode) => void;
    toggleSidebar: () => void;
    togglePropertiesPanel: () => void;
    setTheme: (theme: 'dark' | 'light') => void;
    addNotification: (type: 'info' | 'success' | 'warning' | 'error', message: string) => void;
    removeNotification: (id: string) => void;
}

export const useUIStore = create<UIStore>((set) => ({
    viewMode: 'split',
    sidebarOpen: true,
    propertiesPanelOpen: true,
    theme: 'dark',
    notifications: [],

    setViewMode: (mode) => set({ viewMode: mode }),
    toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
    togglePropertiesPanel: () => set((s) => ({ propertiesPanelOpen: !s.propertiesPanelOpen })),
    setTheme: (theme) => set({ theme }),

    addNotification: (type, message) => set((state) => ({
        notifications: [...state.notifications, { id: Date.now().toString(), type, message }]
    })),

    removeNotification: (id) => set((state) => ({
        notifications: state.notifications.filter(n => n.id !== id)
    })),
}));
