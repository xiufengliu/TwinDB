// Quick Comparison Builder - Simplified UI for building comparison queries
import { useState, useCallback, useRef } from 'react';
import { X, Play, Wand2, ArrowRight, Maximize2, Minimize2 } from 'lucide-react';
import { useWorkspaceStore, useQueryStore, useUIStore } from '../../store';
import './QuickCompare.css';

interface QuickCompareProps {
    onClose: () => void;
}

export default function QuickCompare({ onClose }: QuickCompareProps) {
    const { twins, scenarios } = useWorkspaceStore();
    const { setQuery } = useQueryStore();
    const { addNotification, setViewMode } = useUIStore();

    const [selectedTwin, setSelectedTwin] = useState<string>('');
    const [selectedScenarios, setSelectedScenarios] = useState<string[]>([]);
    const [metric, setMetric] = useState<string>('heat_load');
    const [aggregation, setAggregation] = useState<string>('month');
    const [dateStart, setDateStart] = useState<string>('2018-01-01');
    const [dateEnd, setDateEnd] = useState<string>('2019-01-01');

    // Resize state
    const [size, setSize] = useState({ width: 650, height: 580 });
    const [isResizing, setIsResizing] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);
    const resizeRef = useRef<{ startX: number; startY: number; startW: number; startH: number } | null>(null);

    const metrics = ['heat_load', 'energy', 'temp_supply', 'flow_rate', 'power'];
    const aggregations = ['hour', 'day', 'week', 'month', 'year'];

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

    // Handle resize move (attached globally in JSX)
    const handleResizeMove = useCallback((e: React.MouseEvent) => {
        if (!isResizing || !resizeRef.current) return;
        const newWidth = Math.max(400, resizeRef.current.startW + (e.clientX - resizeRef.current.startX));
        const newHeight = Math.max(350, resizeRef.current.startH + (e.clientY - resizeRef.current.startY));
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

    // Toggle expanded/collapsed
    const toggleExpand = useCallback(() => {
        if (isExpanded) {
            setSize({ width: 650, height: 580 });
        } else {
            setSize({ width: window.innerWidth - 100, height: window.innerHeight - 100 });
        }
        setIsExpanded(!isExpanded);
    }, [isExpanded]);

    // Handle overlay click - only close if not resizing and didn't just finish resizing
    const handleOverlayClick = useCallback((e: React.MouseEvent) => {
        if (e.target === e.currentTarget && !isResizing && !wasResizingRef.current) {
            onClose();
        }
    }, [isResizing, onClose]);

    const handleToggleScenario = (cid: string) => {
        if (selectedScenarios.includes(cid)) {
            setSelectedScenarios(selectedScenarios.filter(s => s !== cid));
        } else {
            setSelectedScenarios([...selectedScenarios, cid]);
        }
    };

    const generateQuery = useCallback(() => {
        if (!selectedTwin) {
            addNotification('error', 'Please select a twin');
            return null;
        }
        if (selectedScenarios.length === 0) {
            addNotification('error', 'Please select at least one scenario');
            return null;
        }

        // Build query
        const selectParts: string[] = ['h.value AS baseline'];
        const fromParts: string[] = [
            `HISTORICAL twin_id='${selectedTwin}'`,
            `  WINDOW '${dateStart}' TO '${dateEnd}'`,
            `  METRIC '${metric}'`,
            `  AGGREGATE BY ${aggregation} AS h`,
        ];

        selectedScenarios.forEach((cid, i) => {
            const alias = `s${i + 1}`;
            const scenario = scenarios.get(cid);
            const label = scenario?.name || cid;

            selectParts.push(`${alias}.value AS ${label.replace(/\s/g, '_')}`);
            fromParts.push(
                `SIMULATE twin_id='${selectedTwin}'`,
                `  SCENARIO ${cid}`,
                `  MODEL 'BuildingHeat_v3'`,
                `  WINDOW '${dateStart}' TO '${dateEnd}'`,
                `  METRIC '${metric}'`,
                `  AGGREGATE BY ${aggregation} AS ${alias}`
            );
        });

        // Add savings calculation
        if (selectedScenarios.length === 1) {
            selectParts.push('(h.value - s1.value) AS saving');
        }

        const query = [
            '-- Auto-generated Comparison Query',
            'TWIN SELECT',
            `  ${selectParts.join(',\n  ')}`,
            'FROM',
            `  ${fromParts.join(',\n  ')}`,
        ].join('\n');

        return query;
    }, [selectedTwin, selectedScenarios, metric, aggregation, dateStart, dateEnd, scenarios, addNotification]);

    const handleGenerate = useCallback(() => {
        const query = generateQuery();
        if (query) {
            setQuery(query);
            addNotification('success', 'Query generated');
            setViewMode('split');
            onClose();
        }
    }, [generateQuery, setQuery, addNotification, setViewMode, onClose]);

    return (
        <div
            className="modal-overlay"
            onClick={handleOverlayClick}
            onMouseMove={isResizing ? handleResizeMove : undefined}
            onMouseUp={isResizing ? handleResizeEnd : undefined}
            onMouseLeave={isResizing ? handleResizeEnd : undefined}
        >
            <div
                className={'modal quick-compare-modal ' + (isResizing ? 'resizing' : '')}
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
                        <Wand2 size={20} />
                        <h3>Quick Comparison Builder</h3>
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

                <div className="modal-body">
                    <div className="compare-flow">
                        {/* Step 1: Select Twin */}
                        <div className="flow-step">
                            <div className="step-number">1</div>
                            <div className="step-content">
                                <label>Select Twin</label>
                                <select value={selectedTwin} onChange={e => setSelectedTwin(e.target.value)}>
                                    <option value="">Choose a twin...</option>
                                    {Array.from(twins.values()).map(twin => (
                                        <option key={twin.tid} value={twin.tid}>{twin.tid}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        <ArrowRight className="flow-arrow" size={20} />

                        {/* Step 2: Select Scenarios */}
                        <div className="flow-step">
                            <div className="step-number">2</div>
                            <div className="step-content">
                                <label>Compare Scenarios</label>
                                <div className="scenario-chips">
                                    {Array.from(scenarios.values())
                                        .filter(s => s.cid !== 'REALITY')
                                        .map(scenario => (
                                            <button
                                                key={scenario.cid}
                                                className={`scenario-chip ${selectedScenarios.includes(scenario.cid) ? 'selected' : ''}`}
                                                onClick={() => handleToggleScenario(scenario.cid)}
                                            >
                                                {scenario.name || scenario.cid}
                                            </button>
                                        ))}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Options */}
                    <div className="compare-options">
                        <div className="option-group">
                            <label>Metric</label>
                            <select value={metric} onChange={e => setMetric(e.target.value)}>
                                {metrics.map(m => (
                                    <option key={m} value={m}>{m}</option>
                                ))}
                            </select>
                        </div>

                        <div className="option-group">
                            <label>Aggregate By</label>
                            <select value={aggregation} onChange={e => setAggregation(e.target.value)}>
                                {aggregations.map(a => (
                                    <option key={a} value={a}>{a}</option>
                                ))}
                            </select>
                        </div>

                        <div className="option-group">
                            <label>Date Range</label>
                            <div className="date-range">
                                <input
                                    type="date"
                                    value={dateStart}
                                    onChange={e => setDateStart(e.target.value)}
                                />
                                <span>to</span>
                                <input
                                    type="date"
                                    value={dateEnd}
                                    onChange={e => setDateEnd(e.target.value)}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Preview */}
                    {selectedTwin && selectedScenarios.length > 0 && (
                        <div className="query-preview">
                            <label>Generated Query</label>
                            <pre>{generateQuery()}</pre>
                        </div>
                    )}
                </div>

                <div className="modal-footer">
                    <button className="btn-secondary" onClick={onClose}>Cancel</button>
                    <button
                        className="btn-primary"
                        onClick={handleGenerate}
                        disabled={!selectedTwin || selectedScenarios.length === 0}
                    >
                        <Play size={16} />
                        <span>Generate & Run</span>
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
