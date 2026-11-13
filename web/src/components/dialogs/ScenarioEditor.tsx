// Scenario Editor - Create and edit scenario configurations with parameter suggestions
import { useState, useCallback } from 'react';
import { X, Save, Plus, Trash2, ChevronRight, HelpCircle, Info } from 'lucide-react';
import { useWorkspaceStore, useUIStore } from '../../store';
import type { Scenario } from '../../types';
import './ScenarioEditor.css';

interface ScenarioEditorProps {
    scenario?: Scenario;
    onClose: () => void;
}

interface RetrofitParam {
    key: string;
    value: string;
}

// Available parameters with descriptions and default values
const availableParams = [
    { key: 'wall_u_before', description: 'Wall U-value before retrofit (W/m²K)', default: '1.2', category: 'Envelope' },
    { key: 'wall_u_after', description: 'Wall U-value after retrofit (W/m²K)', default: '0.18', category: 'Envelope' },
    { key: 'window_u_before', description: 'Window U-value before retrofit (W/m²K)', default: '2.8', category: 'Envelope' },
    { key: 'window_u_after', description: 'Window U-value after retrofit (W/m²K)', default: '1.0', category: 'Envelope' },
    { key: 'roof_u_before', description: 'Roof U-value before retrofit (W/m²K)', default: '0.8', category: 'Envelope' },
    { key: 'roof_u_after', description: 'Roof U-value after retrofit (W/m²K)', default: '0.12', category: 'Envelope' },
    { key: 'floor_u_before', description: 'Floor U-value before retrofit (W/m²K)', default: '0.6', category: 'Envelope' },
    { key: 'floor_u_after', description: 'Floor U-value after retrofit (W/m²K)', default: '0.15', category: 'Envelope' },
    { key: 'infiltration_before', description: 'Air changes per hour before', default: '1.5', category: 'Ventilation' },
    { key: 'infiltration_after', description: 'Air changes per hour after', default: '0.5', category: 'Ventilation' },
    { key: 'ventilation_rate', description: 'Mechanical ventilation rate (L/s/m²)', default: '0.35', category: 'Ventilation' },
    { key: 'heat_recovery_eff', description: 'Heat recovery efficiency (%)', default: '85', category: 'Ventilation' },
    { key: 'heating_efficiency', description: 'Heating system efficiency', default: '0.92', category: 'Systems' },
    { key: 'boiler_efficiency', description: 'Boiler efficiency', default: '0.88', category: 'Systems' },
    { key: 'setpoint_temp', description: 'Indoor temperature setpoint (°C)', default: '21', category: 'Controls' },
    { key: 'setback_temp', description: 'Setback temperature (°C)', default: '18', category: 'Controls' },
    { key: 'internal_gains', description: 'Internal heat gains (W/m²)', default: '5', category: 'Loads' },
    { key: 'occupancy_density', description: 'Occupancy (persons/m²)', default: '0.05', category: 'Loads' },
];

export default function ScenarioEditor({ scenario, onClose }: ScenarioEditorProps) {
    const { addScenario, scenarios } = useWorkspaceStore();
    const { addNotification } = useUIStore();

    const [name, setName] = useState(scenario?.name || '');
    const [cid, setCid] = useState(scenario?.cid || '');
    const [baseScenario, setBaseScenario] = useState(scenario?.base_cid || '');
    const [params, setParams] = useState<RetrofitParam[]>(
        scenario?.cfg
            ? Object.entries(scenario.cfg).map(([key, value]) => ({ key, value: String(value) }))
            : [
                { key: 'wall_u_after', value: '0.20' },
                { key: 'window_u_after', value: '1.0' },
                { key: 'infiltration_after', value: '0.5' },
            ]
    );

    const [showSuggestions, setShowSuggestions] = useState(false);
    const [focusedIndex, setFocusedIndex] = useState<number | null>(null);
    const [filterText, setFilterText] = useState('');
    const [showParamGuide, setShowParamGuide] = useState(false);

    const isEditing = !!scenario;

    // Already used parameters
    const usedParams = new Set(params.map(p => p.key));

    // Filtered suggestions
    const suggestions = availableParams.filter(p =>
        !usedParams.has(p.key) &&
        (filterText === '' ||
            p.key.toLowerCase().includes(filterText.toLowerCase()) ||
            p.description.toLowerCase().includes(filterText.toLowerCase()) ||
            p.category.toLowerCase().includes(filterText.toLowerCase()))
    );

    const handleAddParam = useCallback(() => {
        setParams([...params, { key: '', value: '' }]);
        setFocusedIndex(params.length);
        setShowSuggestions(true);
    }, [params]);

    const handleRemoveParam = useCallback((index: number) => {
        setParams(params.filter((_, i) => i !== index));
    }, [params]);

    const handleParamChange = useCallback((index: number, field: 'key' | 'value', value: string) => {
        const updated = [...params];
        updated[index][field] = value;
        setParams(updated);
        if (field === 'key') {
            setFilterText(value);
        }
    }, [params]);

    const handleSelectSuggestion = useCallback((param: typeof availableParams[0]) => {
        if (focusedIndex !== null) {
            const updated = [...params];
            updated[focusedIndex] = { key: param.key, value: param.default };
            setParams(updated);
        }
        setShowSuggestions(false);
        setFilterText('');
    }, [focusedIndex, params]);

    const handleAddFromGuide = useCallback((param: typeof availableParams[0]) => {
        if (!usedParams.has(param.key)) {
            setParams([...params, { key: param.key, value: param.default }]);
            addNotification('info', 'Added ' + param.key);
        }
    }, [params, usedParams, addNotification]);

    const handleSave = useCallback(() => {
        if (!name.trim() || !cid.trim()) {
            addNotification('error', 'Please provide name and ID');
            return;
        }

        // Build config object
        const cfg: Record<string, unknown> = {};
        params.forEach(p => {
            if (p.key.trim()) {
                const numValue = parseFloat(p.value);
                cfg[p.key] = isNaN(numValue) ? p.value : numValue;
            }
        });

        const newScenario: Scenario = {
            cid,
            name,
            cfg,
            base_cid: baseScenario || undefined,
        };

        addScenario(newScenario);
        addNotification('success', 'Scenario "' + name + '" ' + (isEditing ? 'updated' : 'created'));
        onClose();
    }, [name, cid, baseScenario, params, addScenario, addNotification, isEditing, onClose]);

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal scenario-editor-modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>{isEditing ? 'Edit Scenario' : 'Create Scenario'}</h3>
                    <button className="icon-btn-sm" onClick={onClose}>
                        <X size={18} />
                    </button>
                </div>

                <div className="modal-body">
                    {/* Basic Info */}
                    <div className="form-section">
                        <h4>Basic Information</h4>
                        <div className="form-row">
                            <div className="form-group">
                                <label>Scenario ID</label>
                                <input
                                    type="text"
                                    value={cid}
                                    onChange={e => setCid(e.target.value.replace(/\s/g, '_'))}
                                    placeholder="e.g., deep_retrofit"
                                    disabled={isEditing}
                                />
                                <span className="hint">Unique identifier (no spaces)</span>
                            </div>
                            <div className="form-group">
                                <label>Display Name</label>
                                <input
                                    type="text"
                                    value={name}
                                    onChange={e => setName(e.target.value)}
                                    placeholder="e.g., Deep Retrofit"
                                />
                            </div>
                        </div>

                        <div className="form-group">
                            <label>Base Scenario (Optional)</label>
                            <select value={baseScenario} onChange={e => setBaseScenario(e.target.value)}>
                                <option value="">None - standalone scenario</option>
                                {Array.from(scenarios.values())
                                    .filter(s => s.cid !== cid)
                                    .map(s => (
                                        <option key={s.cid} value={s.cid}>{s.name || s.cid}</option>
                                    ))}
                            </select>
                            <span className="hint">Inherit parameters from another scenario</span>
                        </div>
                    </div>

                    {/* Parameters */}
                    <div className="form-section">
                        <div className="section-header">
                            <h4>Retrofit Parameters</h4>
                            <div className="section-actions">
                                <button
                                    className={'btn-icon help-toggle ' + (showParamGuide ? 'active' : '')}
                                    onClick={() => setShowParamGuide(!showParamGuide)}
                                    title="Show parameter reference"
                                >
                                    <HelpCircle size={16} />
                                </button>
                                <button className="btn-icon" onClick={handleAddParam}>
                                    <Plus size={16} />
                                    <span>Add</span>
                                </button>
                            </div>
                        </div>

                        {/* Parameter Guide */}
                        {showParamGuide && (
                            <div className="param-guide">
                                <div className="param-guide-header">
                                    <Info size={14} />
                                    <span>Available Parameters</span>
                                    <span className="param-guide-hint">Click to add</span>
                                </div>
                                <div className="param-guide-categories">
                                    {['Envelope', 'Ventilation', 'Systems', 'Controls', 'Loads'].map(category => {
                                        const categoryParams = availableParams.filter(p => p.category === category);
                                        return (
                                            <div key={category} className="param-category">
                                                <span className="param-category-name">{category}</span>
                                                <div className="param-category-items">
                                                    {categoryParams.map(p => (
                                                        <button
                                                            key={p.key}
                                                            className={'param-chip ' + (usedParams.has(p.key) ? 'used' : '')}
                                                            onClick={() => handleAddFromGuide(p)}
                                                            disabled={usedParams.has(p.key)}
                                                            title={p.description + ' (default: ' + p.default + ')'}
                                                        >
                                                            {p.key}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        <div className="params-list">
                            {params.map((param, index) => (
                                <div key={index} className="param-row">
                                    <div className="param-key-wrapper">
                                        <input
                                            type="text"
                                            value={param.key}
                                            onChange={e => handleParamChange(index, 'key', e.target.value)}
                                            onFocus={() => {
                                                setFocusedIndex(index);
                                                setFilterText(param.key);
                                                if (!param.key) setShowSuggestions(true);
                                            }}
                                            onBlur={() => {
                                                setTimeout(() => setShowSuggestions(false), 200);
                                            }}
                                            placeholder="Parameter name"
                                            className="param-key"
                                            list={'param-suggestions-' + index}
                                        />
                                        {showSuggestions && focusedIndex === index && suggestions.length > 0 && (
                                            <div className="param-suggestions">
                                                {suggestions.slice(0, 6).map(s => (
                                                    <button
                                                        key={s.key}
                                                        className="suggestion-item"
                                                        onMouseDown={(e) => {
                                                            e.preventDefault();
                                                            handleSelectSuggestion(s);
                                                        }}
                                                    >
                                                        <span className="suggestion-key">{s.key}</span>
                                                        <span className="suggestion-desc">{s.description}</span>
                                                    </button>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                    <ChevronRight size={16} className="param-arrow" />
                                    <input
                                        type="text"
                                        value={param.value}
                                        onChange={e => handleParamChange(index, 'value', e.target.value)}
                                        placeholder="Value"
                                        className="param-value"
                                    />
                                    <button
                                        className="btn-icon-danger"
                                        onClick={() => handleRemoveParam(index)}
                                    >
                                        <Trash2 size={14} />
                                    </button>
                                </div>
                            ))}

                            {params.length === 0 && (
                                <div className="params-empty">
                                    No parameters defined. Click "Add" or use the parameter guide.
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Preview */}
                    <div className="form-section">
                        <h4>Configuration Preview</h4>
                        <pre className="config-preview">
                            {JSON.stringify(
                                {
                                    cid,
                                    name,
                                    base_cid: baseScenario || undefined,
                                    cfg: Object.fromEntries(
                                        params
                                            .filter(p => p.key.trim())
                                            .map(p => [p.key, isNaN(parseFloat(p.value)) ? p.value : parseFloat(p.value)])
                                    ),
                                },
                                null,
                                2
                            )}
                        </pre>
                    </div>
                </div>

                <div className="modal-footer">
                    <button className="btn-secondary" onClick={onClose}>Cancel</button>
                    <button className="btn-primary" onClick={handleSave}>
                        <Save size={16} />
                        <span>{isEditing ? 'Update' : 'Create'} Scenario</span>
                    </button>
                </div>
            </div>
        </div>
    );
}
