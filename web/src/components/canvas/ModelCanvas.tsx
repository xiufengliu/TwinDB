// Model Builder Canvas - React Flow powered workspace with connection support
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    BackgroundVariant,
    MarkerType,
} from '@xyflow/react';
import type { Connection, Node, Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import TwinNode from './TwinNode';
import { useWorkspaceStore, useUIStore, type ConnectionType, type TwinConnection } from '../../store';
import type { Twin, Asset } from '../../types';
import './ModelCanvas.css';

// Custom node types
const nodeTypes = {
    twin: TwinNode,
};

// Connection type configuration with colors and labels
const connectionTypes: { type: ConnectionType; label: string; color: string }[] = [
    { type: 'supplies_heat_to', label: 'Supplies Heat To', color: '#ef4444' },
    { type: 'connects_to', label: 'Connects To', color: '#06b6d4' },
    { type: 'contains', label: 'Contains', color: '#8b5cf6' },
    { type: 'meters', label: 'Meters', color: '#f59e0b' },
    { type: 'controlled_by', label: 'Controlled By', color: '#14b8a6' },
    { type: 'feeds_into', label: 'Feeds Into', color: '#ec4899' },
];

// Convert store data to React Flow nodes
function twinsToNodes(twins: Map<string, Twin>, assets: Map<string, Asset>, selectedId: string | null): Node[] {
    const nodes: Node[] = [];
    let x = 100;
    let y = 100;

    twins.forEach((twin, tid) => {
        const asset = assets.get(twin.aid);
        if (!asset) return;

        nodes.push({
            id: tid,
            type: 'twin',
            position: { x, y },
            data: {
                twin,
                asset,
                hasData: Math.random() > 0.3,
                metrics: Math.random() > 0.5 ? ['heat_load', 'temp_supply'] : [],
                isSelected: tid === selectedId,
            },
            selected: tid === selectedId,
        });

        x += 280;
        if (x > 800) {
            x = 100;
            y += 250;
        }
    });

    return nodes;
}

// Convert store connections to React Flow edges
function connectionsToEdges(connections: TwinConnection[]): Edge[] {
    return connections.map(conn => {
        const typeConfig = connectionTypes.find(t => t.type === conn.type) || connectionTypes[1];
        return {
            id: conn.id,
            source: conn.source,
            target: conn.target,
            type: 'smoothstep',
            animated: true,
            label: typeConfig.label,
            labelStyle: { fontSize: 10, fontWeight: 500 },
            labelBgStyle: { fill: 'var(--color-bg-secondary)', fillOpacity: 0.9 },
            style: { stroke: typeConfig.color, strokeWidth: 2 },
            markerEnd: {
                type: MarkerType.ArrowClosed,
                color: typeConfig.color,
            },
        };
    });
}

export default function ModelCanvas() {
    const { twins, assets, selectedTwinId, selectTwin, connections, addConnection, removeConnection } = useWorkspaceStore();
    const { addNotification } = useUIStore();

    // State for connection type selector
    const [pendingConnection, setPendingConnection] = useState<{ source: string; target: string } | null>(null);
    const [showTypeSelector, setShowTypeSelector] = useState(false);

    // Legend drag state
    const [legendPosition, setLegendPosition] = useState({ x: 16, y: -16 }); // bottom-left offset
    const [isDraggingLegend, setIsDraggingLegend] = useState(false);
    const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

    const initialNodes = useMemo(
        () => twinsToNodes(twins, assets, selectedTwinId),
        [twins, assets, selectedTwinId]
    );

    const initialEdges = useMemo(
        () => connectionsToEdges(connections),
        [connections]
    );

    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    useEffect(() => {
        setNodes(twinsToNodes(twins, assets, selectedTwinId));
    }, [twins, assets, selectedTwinId, setNodes]);

    useEffect(() => {
        setEdges(connectionsToEdges(connections));
    }, [connections, setEdges]);

    // Handle new connection - show type selector
    const onConnect = useCallback(
        (params: Connection) => {
            if (params.source && params.target) {
                // Check if connection already exists
                const exists = connections.some(
                    c => (c.source === params.source && c.target === params.target) ||
                        (c.source === params.target && c.target === params.source)
                );
                if (exists) {
                    addNotification('warning', 'Connection already exists');
                    return;
                }

                setPendingConnection({ source: params.source, target: params.target });
                setShowTypeSelector(true);
            }
        },
        [connections, addNotification]
    );

    // Handle connection type selection
    const handleSelectConnectionType = useCallback((type: ConnectionType) => {
        if (pendingConnection) {
            const newConnection: TwinConnection = {
                id: pendingConnection.source + '_' + pendingConnection.target + '_' + Date.now(),
                source: pendingConnection.source,
                target: pendingConnection.target,
                type,
            };
            addConnection(newConnection);

            const typeConfig = connectionTypes.find(t => t.type === type);
            addNotification('success', 'Created connection: ' + (typeConfig?.label || type));
        }
        setPendingConnection(null);
        setShowTypeSelector(false);
    }, [pendingConnection, addConnection, addNotification]);

    // Handle edge click for deletion
    const onEdgeClick = useCallback(
        (_: React.MouseEvent, edge: Edge) => {
            if (window.confirm('Delete this connection?')) {
                removeConnection(edge.id);
                addNotification('info', 'Connection removed');
            }
        },
        [removeConnection, addNotification]
    );

    const onNodeClick = useCallback(
        (_: React.MouseEvent, node: Node) => {
            selectTwin(node.id);
        },
        [selectTwin]
    );

    const onPaneClick = useCallback(() => {
        selectTwin(null);
        setShowTypeSelector(false);
        setPendingConnection(null);
    }, [selectTwin]);

    return (
        <div className="model-canvas">
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onNodeClick={onNodeClick}
                onEdgeClick={onEdgeClick}
                onPaneClick={onPaneClick}
                nodeTypes={nodeTypes}
                fitView
                minZoom={0.3}
                maxZoom={2}
                defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
                proOptions={{ hideAttribution: true }}
                connectionLineStyle={{ stroke: 'var(--color-accent-primary)', strokeWidth: 2 }}
            >
                <Background
                    variant={BackgroundVariant.Dots}
                    gap={20}
                    size={1}
                    color="var(--color-border)"
                />
                <Controls
                    className="canvas-controls"
                    showInteractive={false}
                />
                <MiniMap
                    className="canvas-minimap"
                    nodeColor={(node) => {
                        const twin = twins.get(node.id);
                        const asset = twin ? assets.get(twin.aid) : null;
                        const assetType = asset?.type as string | undefined;
                        if (assetType === 'building') return 'var(--color-accent-primary)';
                        if (assetType === 'substation') return 'var(--color-accent-warning)';
                        if (assetType === 'heat_plant') return '#ef4444';
                        if (assetType === 'pipeline') return '#06b6d4';
                        return 'var(--color-text-muted)';
                    }}
                    maskColor="rgba(0, 0, 0, 0.7)"
                />
            </ReactFlow>

            {nodes.length === 0 && (
                <div className="canvas-empty">
                    <div className="canvas-empty-content">
                        <span className="canvas-empty-icon">🏗️</span>
                        <h3>No Digital Twins</h3>
                        <p>Drag components from the sidebar or load existing twins from the database.</p>
                    </div>
                </div>
            )}

            {/* Connection Type Selector Modal */}
            {showTypeSelector && pendingConnection && (
                <div className="connection-type-overlay" onClick={() => {
                    setShowTypeSelector(false);
                    setPendingConnection(null);
                }}>
                    <div
                        className="connection-type-selector"
                        onClick={e => e.stopPropagation()}
                    >
                        <div className="selector-header">
                            <h4>Select Connection Type</h4>
                            <span className="selector-hint">
                                {pendingConnection.source.split('_').slice(0, 2).join('_')} → {pendingConnection.target.split('_').slice(0, 2).join('_')}
                            </span>
                        </div>
                        <div className="selector-options">
                            {connectionTypes.map(({ type, label, color }) => (
                                <button
                                    key={type}
                                    className="selector-option"
                                    onClick={() => handleSelectConnectionType(type)}
                                    style={{ '--connection-color': color } as React.CSSProperties}
                                >
                                    <div className="option-indicator" style={{ background: color }} />
                                    <span className="option-label">{label}</span>
                                </button>
                            ))}
                        </div>
                        <button className="selector-cancel" onClick={() => {
                            setShowTypeSelector(false);
                            setPendingConnection(null);
                        }}>
                            Cancel
                        </button>
                    </div>
                </div>
            )}

            {/* Connection Legend - Draggable */}
            <div
                className={'connection-legend ' + (isDraggingLegend ? 'dragging' : '')}
                style={{
                    left: legendPosition.x,
                    bottom: legendPosition.y > 0 ? 'auto' : -legendPosition.y,
                    top: legendPosition.y > 0 ? legendPosition.y : 'auto',
                    cursor: isDraggingLegend ? 'grabbing' : 'grab',
                }}
                onMouseDown={(e) => {
                    e.preventDefault();
                    setIsDraggingLegend(true);
                    const rect = e.currentTarget.getBoundingClientRect();
                    setDragOffset({ x: e.clientX - rect.left, y: e.clientY - rect.top });
                }}
            >
                <span className="legend-title">Connections</span>
                {connectionTypes.slice(0, 4).map(({ type, label, color }) => (
                    <div key={type} className="legend-item">
                        <div className="legend-line" style={{ background: color }} />
                        <span>{label}</span>
                    </div>
                ))}
            </div>

            {/* Global mouse move/up handlers for legend dragging */}
            {isDraggingLegend && (
                <div
                    className="drag-overlay"
                    onMouseMove={(e) => {
                        setLegendPosition({
                            x: e.clientX - dragOffset.x,
                            y: e.clientY - dragOffset.y,
                        });
                    }}
                    onMouseUp={() => setIsDraggingLegend(false)}
                    onMouseLeave={() => setIsDraggingLegend(false)}
                />
            )}
        </div>
    );
}
