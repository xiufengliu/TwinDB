// Results chart component using Recharts
import { useMemo } from 'react';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    BarChart,
    Bar,
} from 'recharts';
import type { QueryResult } from '../../types';
import './ResultsChart.css';

interface ResultsChartProps {
    results: QueryResult;
}

export default function ResultsChart({ results }: ResultsChartProps) {
    // Transform data for Recharts
    const chartData = useMemo(() => {
        return results.data.map(row => {
            const obj: Record<string, unknown> = {};
            results.columns.forEach((col, i) => {
                // Use column name as key, parse numbers
                const val = row[i];
                obj[col] = typeof val === 'string' && !isNaN(Number(val)) ? Number(val) : val;
            });
            return obj;
        });
    }, [results]);

    // Determine which columns to chart
    const numericColumns = useMemo(() => {
        return results.columns.filter((col, i) => {
            if (col.toLowerCase().includes('period') || col.toLowerCase().includes('date')) return false;
            if (col.toLowerCase().includes('pct') || col.toLowerCase().includes('percent')) return false;
            // Check if first data row has numeric value
            const sampleVal = results.data[0]?.[i];
            return typeof sampleVal === 'number' || !isNaN(Number(sampleVal));
        });
    }, [results]);

    // Get x-axis key
    const xAxisKey = useMemo(() => {
        return results.columns.find(c =>
            c.toLowerCase().includes('period') ||
            c.toLowerCase().includes('month') ||
            c.toLowerCase().includes('date')
        ) || results.columns[0];
    }, [results.columns]);

    // Colors for chart series
    const colors = [
        'var(--color-accent-primary)',
        'var(--color-accent-success)',
        'var(--color-accent-warning)',
        'var(--color-accent-secondary)',
        'var(--color-accent-danger)',
    ];

    // Check if this looks like a comparison chart (baseline vs retrofit)
    const isComparison = numericColumns.some(c =>
        c.toLowerCase().includes('baseline') || c.toLowerCase().includes('retrofit')
    );

    // Custom tooltip
    const CustomTooltip = ({ active, payload, label }: any) => {
        if (!active || !payload) return null;

        return (
            <div className="chart-tooltip">
                <div className="chart-tooltip-label">{label}</div>
                {payload.map((entry: any, i: number) => (
                    <div key={i} className="chart-tooltip-row">
                        <span className="chart-tooltip-name" style={{ color: entry.color }}>
                            {entry.name}:
                        </span>
                        <span className="chart-tooltip-value">
                            {typeof entry.value === 'number'
                                ? entry.value.toLocaleString(undefined, { maximumFractionDigits: 0 })
                                : entry.value}
                        </span>
                    </div>
                ))}
            </div>
        );
    };

    if (numericColumns.length === 0) {
        return (
            <div className="chart-empty">
                <span>No numeric data to chart</span>
            </div>
        );
    }

    return (
        <div className="results-chart">
            {isComparison ? (
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                        <XAxis
                            dataKey={xAxisKey}
                            stroke="var(--color-text-muted)"
                            tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                        />
                        <YAxis
                            stroke="var(--color-text-muted)"
                            tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                            tickFormatter={(value) => value >= 1000 ? `${(value / 1000).toFixed(0)}k` : value}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend
                            wrapperStyle={{ paddingTop: 10 }}
                            formatter={(value) => <span style={{ color: 'var(--color-text-secondary)' }}>{value}</span>}
                        />
                        {numericColumns.map((col, i) => (
                            <Bar
                                key={col}
                                dataKey={col}
                                fill={colors[i % colors.length]}
                                radius={[4, 4, 0, 0]}
                            />
                        ))}
                    </BarChart>
                </ResponsiveContainer>
            ) : (
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                        <defs>
                            {numericColumns.map((col, i) => (
                                <linearGradient key={col} id={`gradient-${i}`} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={colors[i % colors.length]} stopOpacity={0.3} />
                                    <stop offset="95%" stopColor={colors[i % colors.length]} stopOpacity={0} />
                                </linearGradient>
                            ))}
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                        <XAxis
                            dataKey={xAxisKey}
                            stroke="var(--color-text-muted)"
                            tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                        />
                        <YAxis
                            stroke="var(--color-text-muted)"
                            tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                            tickFormatter={(value) => value >= 1000 ? `${(value / 1000).toFixed(0)}k` : value}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend
                            wrapperStyle={{ paddingTop: 10 }}
                            formatter={(value) => <span style={{ color: 'var(--color-text-secondary)' }}>{value}</span>}
                        />
                        {numericColumns.map((col, i) => (
                            <Area
                                key={col}
                                type="monotone"
                                dataKey={col}
                                stroke={colors[i % colors.length]}
                                strokeWidth={2}
                                fill={`url(#gradient-${i})`}
                            />
                        ))}
                    </AreaChart>
                </ResponsiveContainer>
            )}
        </div>
    );
}
