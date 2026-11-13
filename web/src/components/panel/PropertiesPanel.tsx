// Properties Panel - Display and edit selected twin/asset details
import { useState, useEffect } from 'react';
import { X, Save, Upload, Settings, Database, ChevronRight } from 'lucide-react';
import { useWorkspaceStore, useUIStore } from '../../store';
import type { Asset } from '../../types';
import ModelConfig from '../dialogs/ModelConfig';
import DataUpload from '../dialogs/DataUpload';
import './PropertiesPanel.css';

export default function PropertiesPanel() {
    const { selectedTwinId, twins, assets, updateAsset, selectTwin } = useWorkspaceStore();
    const { addNotification } = useUIStore();

    const [editedAsset, setEditedAsset] = useState<Asset | null>(null);
    const [activeSection, setActiveSection] = useState<'details' | 'data' | 'model'>('details');
    const [isDirty, setIsDirty] = useState(false);
    const [showModelConfig, setShowModelConfig] = useState(false);
    const [showDataUpload, setShowDataUpload] = useState(false);

    // Get selected twin and asset
    const selectedTwin = selectedTwinId ? twins.get(selectedTwinId) : null;
    const selectedAsset = selectedTwin ? assets.get(selectedTwin.aid) : null;

    // Sync edited state with selection
    useEffect(() => {
        if (selectedAsset) {
            setEditedAsset({ ...selectedAsset });
            setIsDirty(false);
        } else {
            setEditedAsset(null);
        }
    }, [selectedAsset]);

    // Handle field changes
    const handleFieldChange = (field: keyof Asset, value: unknown) => {
        if (!editedAsset) return;
        setEditedAsset({ ...editedAsset, [field]: value });
        setIsDirty(true);
    };

    // Save changes
    const handleSave = async () => {
        if (!editedAsset || !selectedTwin) return;
        await updateAsset(editedAsset.aid, editedAsset);
        setIsDirty(false);
        addNotification('success', 'Changes saved');
    };

    // Close panel
    const handleClose = () => {
        selectTwin(null);
    };

    if (!selectedTwin || !editedAsset) {
        return (
            <div className="properties-panel empty">
                <div className="properties-empty">
                    <Settings size={32} />
                    <span>Select a component to view properties</span>
                </div>
            </div>
        );
    }

    return (
        <div className="properties-panel">
            <div className="properties-header">
                <div className="properties-title">
                    <span className="properties-label">Properties</span>
                    <span className="properties-name">{selectedTwin.tid}</span>
                </div>
                <button className="icon-btn" onClick={handleClose}>
                    <X size={18} />
                </button>
            </div>

            <div className="properties-tabs">
                <button
                    className={'properties-tab ' + (activeSection === 'details' ? 'active' : '')}
                    onClick={() => setActiveSection('details')}
                >
                    Details
                </button>
                <button
                    className={'properties-tab ' + (activeSection === 'data' ? 'active' : '')}
                    onClick={() => setActiveSection('data')}
                >
                    Data
                </button>
                <button
                    className={'properties-tab ' + (activeSection === 'model' ? 'active' : '')}
                    onClick={() => setActiveSection('model')}
                >
                    Model
                </button>
            </div>

            <div className="properties-content">
                {activeSection === 'details' && (
                    <div className="properties-section">
                        <div className="form-group">
                            <label>Twin ID</label>
                            <input type="text" value={selectedTwin.tid} disabled className="input disabled" />
                        </div>

                        <div className="form-group">
                            <label>Asset ID</label>
                            <input type="text" value={editedAsset.aid} disabled className="input disabled" />
                        </div>

                        <div className="form-group">
                            <label>Type</label>
                            <select
                                className="select"
                                value={editedAsset.type}
                                onChange={(e) => handleFieldChange('type', e.target.value)}
                            >
                                <option value="building">Building</option>
                                <option value="substation">Substation</option>
                                <option value="equipment">Equipment</option>
                                <option value="district">District</option>
                            </select>
                        </div>

                        <div className="form-row">
                            <div className="form-group">
                                <label>City</label>
                                <input
                                    type="text"
                                    className="input"
                                    value={editedAsset.city || ''}
                                    onChange={(e) => handleFieldChange('city', e.target.value)}
                                />
                            </div>

                            <div className="form-group">
                                <label>Area (m²)</label>
                                <input
                                    type="number"
                                    className="input"
                                    value={editedAsset.area_m2 || ''}
                                    onChange={(e) => handleFieldChange('area_m2', parseFloat(e.target.value) || 0)}
                                />
                            </div>
                        </div>

                        <div className="form-row">
                            <div className="form-group">
                                <label>Usage Type</label>
                                <select
                                    className="select"
                                    value={editedAsset.usage_type || ''}
                                    onChange={(e) => handleFieldChange('usage_type', e.target.value)}
                                >
                                    <option value="">Select...</option>
                                    <option value="residential">Residential</option>
                                    <option value="office">Office</option>
                                    <option value="school">School</option>
                                    <option value="hospital">Hospital</option>
                                    <option value="retail">Retail</option>
                                    <option value="industrial">Industrial</option>
                                </select>
                            </div>

                            <div className="form-group">
                                <label>Energy Label</label>
                                <select
                                    className="select"
                                    value={editedAsset.energy_label || ''}
                                    onChange={(e) => handleFieldChange('energy_label', e.target.value)}
                                >
                                    <option value="">Select...</option>
                                    <option value="A">A</option>
                                    <option value="B">B</option>
                                    <option value="C">C</option>
                                    <option value="D">D</option>
                                    <option value="E">E</option>
                                    <option value="F">F</option>
                                </select>
                            </div>
                        </div>

                        <div className="form-row">
                            <div className="form-group">
                                <label>Year Built</label>
                                <input
                                    type="number"
                                    className="input"
                                    value={editedAsset.year_built || ''}
                                    onChange={(e) => handleFieldChange('year_built', parseInt(e.target.value) || 0)}
                                />
                            </div>

                            <div className="form-group">
                                <label>Network ID</label>
                                <input
                                    type="text"
                                    className="input"
                                    value={editedAsset.network_id || ''}
                                    onChange={(e) => handleFieldChange('network_id', e.target.value)}
                                />
                            </div>
                        </div>
                    </div>
                )}

                {activeSection === 'data' && (
                    <div className="properties-section">
                        <div className="data-section">
                            <div className="data-header">
                                <Database size={16} />
                                <span>Time-Series Data</span>
                            </div>
                            <div className="data-info">
                                <div className="data-stat">
                                    <span className="data-stat-value">2</span>
                                    <span className="data-stat-label">Metrics</span>
                                </div>
                                <div className="data-stat">
                                    <span className="data-stat-value">8,760</span>
                                    <span className="data-stat-label">Points</span>
                                </div>
                            </div>
                            <button className="btn-secondary" onClick={() => setShowDataUpload(true)}>
                                <Upload size={14} />
                                <span>Upload Data</span>
                            </button>
                        </div>

                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-name">heat_load</span>
                                <span className="metric-range">Jan 2018 - Dec 2018</span>
                                <ChevronRight size={14} />
                            </div>
                            <div className="metric-item">
                                <span className="metric-name">temp_supply</span>
                                <span className="metric-range">Jan 2018 - Dec 2018</span>
                                <ChevronRight size={14} />
                            </div>
                        </div>
                    </div>
                )}

                {activeSection === 'model' && (
                    <div className="properties-section">
                        <div className="model-card">
                            <div className="model-header">
                                <span className="model-name">BuildingHeat_v3</span>
                                <span className="model-type badge">phys</span>
                            </div>
                            <div className="model-info">
                                <div className="model-row">
                                    <span>Scope</span>
                                    <span>building</span>
                                </div>
                                <div className="model-row">
                                    <span>Version</span>
                                    <span>3.0</span>
                                </div>
                            </div>
                        </div>

                        <button
                            className="btn-secondary full-width"
                            onClick={() => setShowModelConfig(true)}
                        >
                            <Settings size={14} />
                            <span>Configure Model</span>
                        </button>
                    </div>
                )}
            </div>

            {isDirty && (
                <div className="properties-footer">
                    <button className="btn-primary" onClick={handleSave}>
                        <Save size={14} />
                        <span>Save Changes</span>
                    </button>
                </div>
            )}

            {/* Model Configuration Dialog */}
            {showModelConfig && selectedTwin && (
                <ModelConfig
                    modelId="BuildingHeat_v3"
                    twinId={selectedTwin.tid}
                    onClose={() => setShowModelConfig(false)}
                />
            )}

            {/* Data Upload Dialog */}
            {showDataUpload && selectedTwin && (
                <DataUpload
                    twinId={selectedTwin.tid}
                    onClose={() => setShowDataUpload(false)}
                />
            )}
        </div>
    );
}

