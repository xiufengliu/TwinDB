// Main App Layout - Enhanced with all dialogs integrated
import { useEffect, useCallback, useState, type DragEvent } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import {
  Database, Layers, Terminal, PanelLeftClose, PanelRightClose, Plus,
  Building2, Zap, Wrench, Globe, Trash2, GripVertical,
  Wand2, Settings, Keyboard, Download, Upload, Search, HelpCircle,
  Factory, GitBranch, Cylinder, Activity, Gauge, Thermometer,
  ChevronUp, ChevronDown
} from 'lucide-react';
import ModelCanvas from './components/canvas/ModelCanvas';
import Console from './components/console/Console';
import PropertiesPanel from './components/panel/PropertiesPanel';
import ToastContainer from './components/ui/Toast';
import DataUpload from './components/dialogs/DataUpload';
import ScenarioEditor from './components/dialogs/ScenarioEditor';
import QuickCompare from './components/dialogs/QuickCompare';
import KeyboardShortcuts from './components/dialogs/KeyboardShortcuts';
import { useWorkspaceStore, useUIStore } from './store';
import { api } from './services/api';
import type { Twin, Asset } from './types';
import './App.css';

// Asset type configurations - Comprehensive district heating components
const assetTypes = [
  // Core Infrastructure
  { type: 'building', label: 'Building', icon: Building2, color: 'var(--color-accent-primary)', category: 'consumers' },
  { type: 'substation', label: 'Substation', icon: Zap, color: 'var(--color-accent-warning)', category: 'network' },
  { type: 'district', label: 'District', icon: Globe, color: 'var(--color-accent-secondary)', category: 'aggregation' },
  // Production & Distribution
  { type: 'heat_plant', label: 'Heat Plant', icon: Factory, color: '#ef4444', category: 'production' },
  { type: 'pipeline', label: 'Pipeline', icon: GitBranch, color: '#06b6d4', category: 'network' },
  { type: 'storage', label: 'Storage Tank', icon: Cylinder, color: '#8b5cf6', category: 'storage' },
  // Control & Monitoring
  { type: 'pump_station', label: 'Pump Station', icon: Activity, color: '#14b8a6', category: 'network' },
  { type: 'meter', label: 'Meter', icon: Gauge, color: '#f59e0b', category: 'monitoring' },
  { type: 'heat_exchanger', label: 'Heat Exchanger', icon: Thermometer, color: '#ec4899', category: 'network' },
  { type: 'equipment', label: 'Equipment', icon: Wrench, color: 'var(--color-text-muted)', category: 'other' },
];

export default function App() {
  const {
    twins, assets, scenarios, setTwins, setAssets, setScenarios, setModels,
    selectedTwinId, addTwin, removeTwin, selectTwin
  } = useWorkspaceStore();
  const {
    viewMode, setViewMode, sidebarOpen, toggleSidebar,
    propertiesPanelOpen, addNotification
  } = useUIStore();

  const [isDragOver, setIsDragOver] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Dialog states
  const [showNewTwinModal, setShowNewTwinModal] = useState(false);
  const [showDataUpload, setShowDataUpload] = useState(false);
  const [showScenarioEditor, setShowScenarioEditor] = useState(false);
  const [showQuickCompare, setShowQuickCompare] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);

  // Panel collapse states
  const [workspaceCollapsed, setWorkspaceCollapsed] = useState(false);
  const [consoleCollapsed, setConsoleCollapsed] = useState(false);

  // Load initial data
  useEffect(() => {
    const loadData = async () => {
      const [twinsData, assetsData, scenariosData, modelsData] = await Promise.all([
        api.fetchTwins(),
        api.fetchAssets(),
        api.fetchScenarios(),
        api.fetchModels(),
      ]);
      setTwins(twinsData);
      setAssets(assetsData);
      setScenarios(scenariosData);
      setModels(modelsData);
      addNotification('success', 'Loaded ' + twinsData.length + ' twins');
    };
    loadData();
  }, [setTwins, setAssets, setScenarios, setModels, addNotification]);

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // ? key for shortcuts
      if (e.key === '?' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        setShowShortcuts(true);
      }
      // Ctrl/Cmd + number for view modes
      if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '3') {
        e.preventDefault();
        const modes = ['workspace', 'console', 'split'] as const;
        setViewMode(modes[parseInt(e.key) - 1]);
      }
      // Ctrl/Cmd + B for sidebar
      if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
        e.preventDefault();
        toggleSidebar();
      }
      // Delete for removing selected twin
      if (e.key === 'Delete' && selectedTwinId && !isInputFocused()) {
        e.preventDefault();
        handleDeleteTwin();
      }
      // Escape to deselect
      if (e.key === 'Escape') {
        selectTwin(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setViewMode, toggleSidebar, selectedTwinId, selectTwin]);

  // Check if input is focused
  const isInputFocused = () => {
    const tag = document.activeElement?.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
  };

  // Handle drag start from sidebar
  const handleDragStart = useCallback((e: DragEvent, assetType: string) => {
    e.dataTransfer.setData('application/twindb-asset', assetType);
    e.dataTransfer.effectAllowed = 'copy';
  }, []);

  // Handle drop on canvas
  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const assetType = e.dataTransfer.getData('application/twindb-asset');
    if (!assetType) return;

    createNewTwin(assetType);
  }, []);

  // Create new twin helper
  const createNewTwin = useCallback((assetType: string) => {
    const timestamp = Date.now();
    const newAid = 'A' + timestamp;
    const newTid = 'Twin_' + assetType.charAt(0).toUpperCase() + timestamp;

    const newAsset: Asset = {
      aid: newAid,
      type: assetType as Asset['type'],
      city: 'Copenhagen',
      area_m2: 2000,
      usage_type: assetType === 'building' ? 'residential' : undefined,
      year_built: 2020,
      energy_label: 'C',
      network_id: 'DH_CPH_01',
      attrs: {},
    };

    const newTwin: Twin = {
      tid: newTid,
      aid: newAid,
      twin_type: assetType,
      metadata: { resolution: 'hourly' },
    };

    addTwin(newTwin, newAsset);
    selectTwin(newTid);
    addNotification('success', 'Created new ' + assetType + ': ' + newTid);
  }, [addTwin, selectTwin, addNotification]);

  // Handle delete twin
  const handleDeleteTwin = useCallback(() => {
    if (selectedTwinId) {
      const twinName = selectedTwinId;
      removeTwin(selectedTwinId);
      addNotification('info', 'Deleted ' + twinName);
    }
  }, [selectedTwinId, removeTwin, addNotification]);

  // Export workspace
  const handleExportWorkspace = useCallback(() => {
    const data = {
      twins: Array.from(twins.values()),
      assets: Array.from(assets.values()),
      scenarios: Array.from(scenarios.values()),
      exportedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'twindb_workspace_' + Date.now() + '.json';
    a.click();
    URL.revokeObjectURL(url);
    addNotification('success', 'Workspace exported');
  }, [twins, assets, scenarios, addNotification]);

  // Filter twins by search
  const filteredTwins = Array.from(twins.values()).filter(twin =>
    searchQuery === '' ||
    twin.tid.toLowerCase().includes(searchQuery.toLowerCase()) ||
    twin.twin_type.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="logo">
            <Database size={20} />
            <span className="logo-text">TwinDB</span>
            <span className="logo-badge">BETA</span>
          </div>
          <nav className="header-nav">
            <button
              className={'nav-btn ' + (viewMode === 'workspace' ? 'active' : '')}
              onClick={() => setViewMode('workspace')}
              title="Workspace Only (Ctrl+1)"
            >
              <Layers size={16} />
              <span>Workspace</span>
            </button>
            <button
              className={'nav-btn ' + (viewMode === 'console' ? 'active' : '')}
              onClick={() => setViewMode('console')}
              title="Console Only (Ctrl+2)"
            >
              <Terminal size={16} />
              <span>Console</span>
            </button>
            <button
              className={'nav-btn ' + (viewMode === 'split' ? 'active' : '')}
              onClick={() => setViewMode('split')}
              title="Split View (Ctrl+3)"
            >
              <GripVertical size={16} />
              <span>Split</span>
            </button>
          </nav>
        </div>

        <div className="header-right">
          <button
            className="header-btn"
            onClick={() => setShowQuickCompare(true)}
            title="Quick Comparison Builder"
          >
            <Wand2 size={16} />
            <span>Compare</span>
          </button>
          <button
            className="header-btn"
            onClick={() => setShowScenarioEditor(true)}
            title="Create Scenario"
          >
            <Settings size={16} />
            <span>Scenario</span>
          </button>
          <div className="header-divider" />
          {selectedTwinId && (
            <>
              <button
                className="header-btn"
                onClick={() => setShowDataUpload(true)}
                title="Upload Data to Selected Twin"
              >
                <Upload size={16} />
              </button>
              <button
                className="header-btn danger"
                onClick={handleDeleteTwin}
                title="Delete Selected Twin"
              >
                <Trash2 size={16} />
              </button>
            </>
          )}
          <button className="header-btn" onClick={handleExportWorkspace} title="Export Workspace">
            <Download size={16} />
          </button>
          <button
            className="header-btn"
            onClick={() => setShowShortcuts(true)}
            title="Keyboard Shortcuts (?)"
          >
            <Keyboard size={16} />
          </button>
          <button className="header-btn primary" onClick={() => setShowNewTwinModal(true)}>
            <Plus size={16} />
            <span>New Twin</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {/* Sidebar Toggle */}
        {!sidebarOpen && (
          <button className="sidebar-toggle left" onClick={toggleSidebar} title="Toggle Sidebar (Ctrl+B)">
            <PanelLeftClose size={16} />
          </button>
        )}

        {/* Sidebar */}
        {sidebarOpen && (
          <aside className="app-sidebar">
            <div className="sidebar-header">
              <span>Asset Library</span>
              <button className="icon-btn-sm" onClick={toggleSidebar}>
                <PanelLeftClose size={14} />
              </button>
            </div>
            <div className="sidebar-content">
              <div className="asset-category">
                <span className="category-label">Drag to Canvas</span>
                <div className="asset-items">
                  {assetTypes.map(({ type, label, icon: Icon, color }) => (
                    <div
                      key={type}
                      className="asset-item"
                      draggable
                      onDragStart={(e) => handleDragStart(e, type)}
                    >
                      <div className="asset-icon-wrapper" style={{ background: color + '20', color }}>
                        <Icon size={18} />
                      </div>
                      <span>{label}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="asset-category">
                <div className="category-header">
                  <span className="category-label">Active Twins ({twins.size})</span>
                </div>
                <div className="search-box">
                  <Search size={14} />
                  <input
                    type="text"
                    placeholder="Filter twins..."
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                  />
                </div>
                <div className="twin-list">
                  {filteredTwins.map(twin => {
                    const asset = assets.get(twin.aid);
                    const AssetIcon = assetTypes.find(t => t.type === asset?.type)?.icon || Building2;
                    return (
                      <button
                        key={twin.tid}
                        className={'twin-item ' + (selectedTwinId === twin.tid ? 'selected' : '')}
                        onClick={() => selectTwin(twin.tid)}
                      >
                        <AssetIcon size={14} />
                        <span>{twin.tid}</span>
                      </button>
                    );
                  })}
                  {filteredTwins.length === 0 && (
                    <div className="twin-list-empty">No twins found</div>
                  )}
                </div>
              </div>

              <div className="asset-category">
                <div className="category-header">
                  <span className="category-label">Scenarios ({scenarios.size})</span>
                  <button
                    className="btn-icon-sm"
                    onClick={() => setShowScenarioEditor(true)}
                    title="Create Scenario"
                  >
                    <Plus size={12} />
                  </button>
                </div>
                <div className="scenario-items">
                  {Array.from(scenarios.values()).map(scenario => (
                    <div key={scenario.cid} className="scenario-item">
                      <span className={'scenario-dot ' + (scenario.cid === 'REALITY' ? 'reality' : '')}></span>
                      <span>{scenario.name || scenario.cid}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="sidebar-footer">
              <button
                className="help-btn"
                onClick={() => setShowShortcuts(true)}
              >
                <HelpCircle size={14} />
                <span>Help & Shortcuts</span>
              </button>
            </div>
          </aside>
        )}

        {/* Content Area with Resizable Panels */}
        <div
          className={'app-content ' + (isDragOver ? 'drag-over' : '')}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
        >
          {viewMode === 'workspace' && (
            <div className="content-workspace">
              <ModelCanvas />
            </div>
          )}

          {viewMode === 'console' && (
            <div className="content-console">
              <Console />
            </div>
          )}

          {viewMode === 'split' && (
            <PanelGroup direction="vertical" className="content-split">
              {/* Workspace Panel */}
              <Panel
                defaultSize={workspaceCollapsed ? 5 : 50}
                minSize={workspaceCollapsed ? 5 : 15}
                collapsible={true}
                collapsedSize={5}
              >
                <div className="collapsible-panel">
                  <div className="panel-header" onClick={() => setWorkspaceCollapsed(!workspaceCollapsed)}>
                    <div className="panel-title">
                      <Layers size={14} />
                      <span>Model Workspace</span>
                    </div>
                    <button className="panel-collapse-btn">
                      {workspaceCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
                    </button>
                  </div>
                  {!workspaceCollapsed && (
                    <div className="panel-content">
                      <ModelCanvas />
                    </div>
                  )}
                </div>
              </Panel>

              <PanelResizeHandle className="resize-handle" />

              {/* Console Panel */}
              <Panel
                defaultSize={consoleCollapsed ? 5 : 50}
                minSize={consoleCollapsed ? 5 : 15}
                collapsible={true}
                collapsedSize={5}
              >
                <div className="collapsible-panel">
                  <div className="panel-header" onClick={() => setConsoleCollapsed(!consoleCollapsed)}>
                    <div className="panel-title">
                      <Terminal size={14} />
                      <span>TwinQL Console</span>
                    </div>
                    <button className="panel-collapse-btn">
                      {consoleCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
                    </button>
                  </div>
                  {!consoleCollapsed && (
                    <div className="panel-content">
                      <Console />
                    </div>
                  )}
                </div>
              </Panel>
            </PanelGroup>
          )}

          {/* Drop overlay */}
          {isDragOver && (
            <div className="drop-overlay">
              <div className="drop-message">
                <Plus size={32} />
                <span>Drop to create new twin</span>
              </div>
            </div>
          )}
        </div>

        {/* Properties Panel */}
        {propertiesPanelOpen && selectedTwinId && <PropertiesPanel />}

        {/* Properties Panel Toggle */}
        {!propertiesPanelOpen && selectedTwinId && (
          <button className="sidebar-toggle right" onClick={() => { }}>
            <PanelRightClose size={16} />
          </button>
        )}
      </main>

      {/* Toast Notifications */}
      <ToastContainer />

      {/* New Twin Modal */}
      {showNewTwinModal && (
        <div className="modal-overlay" onClick={() => setShowNewTwinModal(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Create New Twin</h3>
              <button className="icon-btn-sm" onClick={() => setShowNewTwinModal(false)}>×</button>
            </div>
            <div className="modal-body">
              <p className="modal-hint">
                Select a component type to create a new digital twin:
              </p>
              <div className="modal-asset-grid">
                {assetTypes.map(({ type, label, icon: Icon, color }) => (
                  <button
                    key={type}
                    className="modal-asset-btn"
                    onClick={() => {
                      createNewTwin(type);
                      setShowNewTwinModal(false);
                    }}
                  >
                    <div className="asset-icon-wrapper" style={{ background: color + '20', color }}>
                      <Icon size={24} />
                    </div>
                    <span>{label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Data Upload Dialog */}
      {showDataUpload && selectedTwinId && (
        <DataUpload
          twinId={selectedTwinId}
          onClose={() => setShowDataUpload(false)}
        />
      )}

      {/* Scenario Editor Dialog */}
      {showScenarioEditor && (
        <ScenarioEditor onClose={() => setShowScenarioEditor(false)} />
      )}

      {/* Quick Compare Dialog */}
      {showQuickCompare && (
        <QuickCompare onClose={() => setShowQuickCompare(false)} />
      )}

      {/* Keyboard Shortcuts Dialog */}
      {showShortcuts && (
        <KeyboardShortcuts onClose={() => setShowShortcuts(false)} />
      )}
    </div>
  );
}
