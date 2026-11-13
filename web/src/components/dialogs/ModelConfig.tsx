// Model Configuration Dialog
import { useState, useCallback, useRef } from 'react';
import { X, Save, Plus, Trash2, Settings, Play, Maximize2, Minimize2 } from 'lucide-react';
import { useUIStore } from '../../store';
import './ModelConfig.css';

interface ModelConfigProps {
    modelId: string;
    twinId: string;
    onClose: () => void;
}

interface ModelParam {
    name: string;
    value: string;
    type: 'number' | 'string' | 'boolean';
    description: string;
}

// Default parameters for BuildingHeat_v3 model
const defaultParams: ModelParam[] = [
    { name: 'wall_u_before', value: '1.2', type: 'number', description: 'Wall U-value before retrofit (W/m²K)' },
    { name: 'wall_u_after', value: '0.18', type: 'number', description: 'Wall U-value after retrofit (W/m²K)' },
    { name: 'window_u_before', value: '2.8', type: 'number', description: 'Window U-value before retrofit (W/m²K)' },
    { name: 'window_u_after', value: '1.0', type: 'number', description: 'Window U-value after retrofit (W/m²K)' },
    { name: 'infiltration_before', value: '1.5', type: 'number', description: 'Air changes per hour before' },
    { name: 'infiltration_after', value: '0.5', type: 'number', description: 'Air changes per hour after' },
    { name: 'heating_efficiency', value: '0.92', type: 'number', description: 'Heating system efficiency' },
    { name: 'setpoint_temp', value: '21', type: 'number', description: 'Indoor temperature setpoint (°C)' },
];

export default function ModelConfig({ modelId, twinId, onClose }: ModelConfigProps) {
    const { addNotification } = useUIStore();
    const [params, setParams] = useState<ModelParam[]>(defaultParams);
    const [activeTab, setActiveTab] = useState<'params' | 'info'>('params');

    // Resize state
    const [size, setSize] = useState({ width: 700, height: 650 });
    const [isResizing, setIsResizing] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);
    const resizeRef = useRef<{ startX: number; startY: number; startW: number; startH: number } | null>(null);

    const handleParamChange = useCallback((index: number, value: string) => {
        const updated = [...params];
        updated[index].value = value;
        setParams(updated);
    }, [params]);

    const handleAddParam = useCallback(() => {
        setParams([...params, { name: '', value: '', type: 'number', description: '' }]);
    }, [params]);

    const handleRemoveParam = useCallback((index: number) => {
        setParams(params.filter((_, i) => i !== index));
    }, [params]);

    const handleSave = useCallback(() => {
        addNotification('success', 'Model configuration saved for ' + twinId);
        onClose();
    }, [twinId, addNotification, onClose]);

    const handleRunSimulation = useCallback(() => {
        addNotification('info', 'Starting simulation with current parameters...');
        // In real implementation, this would trigger the backend simulation
        setTimeout(() => {
            addNotification('success', 'Simulation completed for ' + twinId);
        }, 1500);
    }, [twinId, addNotification]);

    // Handle resize start
    const handleResizeStart = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        setIsResizing(true);
        resizeRef.current = {
            startX: e.clientX,
            startY: e.clientY,
            startW: size.width,
            startH: size.height,
        };
    }, [size]);

    // Handle resize move
    const handleResizeMove = useCallback((e: React.MouseEvent) => {
        if (!isResizing || !resizeRef.current) return;
        const newWidth = Math.max(500, resizeRef.current.startW + (e.clientX - resizeRef.current.startX));
        const newHeight = Math.max(400, resizeRef.current.startH + (e.clientY - resizeRef.current.startY));
        setSize({ width: Math.min(newWidth, window.innerWidth - 100), height: Math.min(newHeight, window.innerHeight - 100) });
    }, [isResizing]);

    // Ref to track if we just finished resizing (prevents click from closing)
    const wasResizingRef = useRef(false);

    // Handle resize end
    const handleResizeEnd = useCallback(() => {
        wasResizingRef.current = true;
        setIsResizing(false);
        resizeRef.current = null;
        // Reset after a short delay to allow the click event to check it
        setTimeout(() => {
            wasResizingRef.current = false;
        }, 100);
    }, []);

    // Handle overlay click - only close if not resizing and didn't just finish resizing
    const handleOverlayClick = useCallback((e: React.MouseEvent) => {
        if (e.target === e.currentTarget && !isResizing && !wasResizingRef.current) {
            onClose();
        }
    }, [isResizing, onClose]);

    // Toggle expanded/collapsed
    const toggleExpand = useCallback(() => {
        if (isExpanded) {
            setSize({ width: 700, height: 650 });
        } else {
            setSize({ width: window.innerWidth - 100, height: window.innerHeight - 100 });
        }
        setIsExpanded(!isExpanded);
    }, [isExpanded]);

    return (
        <div
            className="modal-overlay"
            onClick={handleOverlayClick}
            onMouseMove={isResizing ? handleResizeMove : undefined}
            onMouseUp={isResizing ? handleResizeEnd : undefined}
            onMouseLeave={isResizing ? handleResizeEnd : undefined}
        >
            <div
                className={'modal model-config-modal ' + (isResizing ? 'resizing' : '')}
                onClick={e => e.stopPropagation()}
                style={{
                    width: size.width,
                    height: size.height,
                    maxWidth: 'none',
                    maxHeight: 'none',
                }}
            >
                <div className="modal-header">
                    <div className="modal-title">
                        <Settings size={20} />
                        <div>
                            <h3>Configure Model</h3>
                            <span className="modal-subtitle">{modelId} • {twinId}</span>
                        </div>
                    </div>
                    <div className="modal-header-actions">
                        <button
                            className="icon-btn-sm"
                            onClick={toggleExpand}
                            title={isExpanded ? 'Restore size' : 'Maximize'}
                        >
                            {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                        </button>
                        <button className="icon-btn-sm" onClick={onClose}>
                            <X size={18} />
                        </button>
                    </div>
                </div>

                <div className="config-tabs">
                    <button
                        className={'config-tab ' + (activeTab === 'params' ? 'active' : '')}
                        onClick={() => setActiveTab('params')}
                    >
                        Parameters
                    </button>
                    <button
                        className={'config-tab ' + (activeTab === 'info' ? 'active' : '')}
                        onClick={() => setActiveTab('info')}
                    >
                        Model Info
                    </button>
                </div>

                <div className="modal-body">
                    {activeTab === 'params' && (
                        <>
                            <div className="params-header">
                                <span className="params-count">{params.length} parameters</span>
                                <button className="btn-icon" onClick={handleAddParam}>
                                    <Plus size={14} />
                                    <span>Add</span>
                                </button>
                            </div>

                            <div className="params-grid">
                                {params.map((param, index) => (
                                    <div key={index} className="param-card">
                                        <div className="param-header">
                                            <input
                                                type="text"
                                                className="param-name"
                                                value={param.name}
                                                onChange={e => {
                                                    const updated = [...params];
                                                    updated[index].name = e.target.value;
                                                    setParams(updated);
                                                }}
                                                placeholder="Parameter name"
                                            />
                                            <button
                                                className="btn-icon-danger"
                                                onClick={() => handleRemoveParam(index)}
                                            >
                                                <Trash2 size={12} />
                                            </button>
                                        </div>
                                        <input
                                            type="text"
                                            className="param-value"
                                            value={param.value}
                                            onChange={e => handleParamChange(index, e.target.value)}
                                            placeholder="Value"
                                        />
                                        <span className="param-desc">{param.description}</span>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}

                    {activeTab === 'info' && (
                        <div className="model-info-content">
                            <div className="info-section">
                                <h4>Model Overview</h4>
                                <div className="info-grid">
                                    <div className="info-item">
                                        <span className="info-label">Name</span>
                                        <span className="info-value">{modelId}</span>
                                    </div>
                                    <div className="info-item">
                                        <span className="info-label">Type</span>
                                        <span className="info-value">Physics-based (5R1C)</span>
                                    </div>
                                    <div className="info-item">
                                        <span className="info-label">Version</span>
                                        <span className="info-value">3.0</span>
                                    </div>
                                    <div className="info-item">
                                        <span className="info-label">Scope</span>
                                        <span className="info-value">Building</span>
                                    </div>
                                </div>
                            </div>

                            <div className="info-section">
                                <h4>Description</h4>
                                <p className="info-description">
                                    A 5-resistor 1-capacitor (5R1C) thermal network model for building heat demand simulation.
                                    This physics-based model calculates hourly heat load based on building envelope properties,
                                    weather data, and internal gains.
                                </p>
                            </div>

                            <div className="info-section">
                                <h4>Inputs</h4>
                                <div className="info-tags">
                                    <span className="info-tag">Weather (temperature, radiation)</span>
                                    <span className="info-tag">Building geometry (area, volume)</span>
                                    <span className="info-tag">Envelope properties (U-values)</span>
                                    <span className="info-tag">Usage patterns</span>
                                </div>
                            </div>

                            <div className="info-section">
                                <h4>Outputs</h4>
                                <div className="info-tags">
                                    <span className="info-tag output">heat_load (kWh)</span>
                                    <span className="info-tag output">indoor_temp (°C)</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                <div className="modal-footer">
                    <button className="btn-secondary" onClick={onClose}>Cancel</button>
                    <button className="btn-secondary" onClick={handleRunSimulation}>
                        <Play size={14} />
                        <span>Run Simulation</span>
                    </button>
                    <button className="btn-primary" onClick={handleSave}>
                        <Save size={14} />
                        <span>Save Configuration</span>
                    </button>
                </div>

                {/* Resize Handle */}
                <div
                    className="resize-handle"
                    onMouseDown={handleResizeStart}
                    title="Drag to resize"
                />
            </div>
        </div>
    );
}
