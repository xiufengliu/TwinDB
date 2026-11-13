// Custom TwinDB canvas node for React Flow
import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Building2, Zap, Box, Network } from 'lucide-react';
import type { Twin, Asset } from '../../types';
import './TwinNode.css';

interface TwinNodeData {
    twin: Twin;
    asset: Asset;
    hasData: boolean;
    metrics: string[];
    isSelected: boolean;
}

interface TwinNodeProps {
    data: TwinNodeData;
    selected: boolean;
}

const getIcon = (type: string) => {
    switch (type) {
        case 'building': return Building2;
        case 'substation': return Zap;
        case 'district': return Network;
        default: return Box;
    }
};

const getTypeColor = (type: string) => {
    switch (type) {
        case 'building': return 'var(--color-accent-primary)';
        case 'substation': return 'var(--color-accent-warning)';
        case 'district': return 'var(--color-accent-secondary)';
        default: return 'var(--color-text-muted)';
    }
};

function TwinNode({ data, selected }: TwinNodeProps) {
    const Icon = getIcon(data.asset.type);
    const color = getTypeColor(data.asset.type);

    return (
        <div className={`twin-node ${selected ? 'selected' : ''}`}>
            <Handle type="target" position={Position.Top} className="twin-handle" />

            <div className="twin-node-header" style={{ borderColor: color }}>
                <div className="twin-node-icon" style={{ background: `${color}20`, color }}>
                    <Icon size={18} />
                </div>
                <div className="twin-node-title">
                    <span className="twin-node-name">{data.twin.tid}</span>
                    <span className="twin-node-type">{data.asset.type}</span>
                </div>
            </div>

            <div className="twin-node-body">
                <div className="twin-node-row">
                    <span className="twin-node-label">Asset</span>
                    <span className="twin-node-value">{data.asset.aid}</span>
                </div>
                {data.asset.city && (
                    <div className="twin-node-row">
                        <span className="twin-node-label">City</span>
                        <span className="twin-node-value">{data.asset.city}</span>
                    </div>
                )}
                {data.asset.area_m2 && (
                    <div className="twin-node-row">
                        <span className="twin-node-label">Area</span>
                        <span className="twin-node-value">{data.asset.area_m2.toLocaleString()} m²</span>
                    </div>
                )}
                {data.asset.energy_label && (
                    <div className="twin-node-row">
                        <span className="twin-node-label">Energy</span>
                        <span className={`twin-node-badge label-${data.asset.energy_label.toLowerCase()}`}>
                            {data.asset.energy_label}
                        </span>
                    </div>
                )}
            </div>

            <div className="twin-node-footer">
                <span className={`twin-node-status ${data.hasData ? 'has-data' : 'no-data'}`}>
                    {data.hasData ? '● Data Available' : '○ No Data'}
                </span>
                {data.metrics.length > 0 && (
                    <span className="twin-node-metrics">{data.metrics.length} metrics</span>
                )}
            </div>

            <Handle type="source" position={Position.Bottom} className="twin-handle" />
        </div>
    );
}

export default memo(TwinNode);
