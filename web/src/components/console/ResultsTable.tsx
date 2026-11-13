// Results table component with sorting and filtering
import { useState, useMemo } from 'react';
import { ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';
import type { QueryResult } from '../../types';
import './ResultsTable.css';

interface ResultsTableProps {
    results: QueryResult;
}

type SortDir = 'asc' | 'desc' | null;

export default function ResultsTable({ results }: ResultsTableProps) {
    const [sortCol, setSortCol] = useState<number | null>(null);
    const [sortDir, setSortDir] = useState<SortDir>(null);

    // Sort data
    const sortedData = useMemo(() => {
        if (sortCol === null || sortDir === null) return results.data;

        return [...results.data].sort((a, b) => {
            const aVal = a[sortCol];
            const bVal = b[sortCol];

            // Handle numeric sorting
            const aNum = typeof aVal === 'number' ? aVal : parseFloat(aVal as string);
            const bNum = typeof bVal === 'number' ? bVal : parseFloat(bVal as string);

            if (!isNaN(aNum) && !isNaN(bNum)) {
                return sortDir === 'asc' ? aNum - bNum : bNum - aNum;
            }

            // String sorting
            const aStr = String(aVal);
            const bStr = String(bVal);
            return sortDir === 'asc' ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
        });
    }, [results.data, sortCol, sortDir]);

    // Handle column header click
    const handleSort = (colIndex: number) => {
        if (sortCol === colIndex) {
            // Cycle: asc -> desc -> none
            if (sortDir === 'asc') setSortDir('desc');
            else if (sortDir === 'desc') { setSortCol(null); setSortDir(null); }
            else setSortDir('asc');
        } else {
            setSortCol(colIndex);
            setSortDir('asc');
        }
    };

    // Get sort icon
    const getSortIcon = (colIndex: number) => {
        if (sortCol !== colIndex) return <ArrowUpDown size={12} className="sort-icon-inactive" />;
        if (sortDir === 'asc') return <ArrowUp size={12} className="sort-icon-active" />;
        if (sortDir === 'desc') return <ArrowDown size={12} className="sort-icon-active" />;
        return <ArrowUpDown size={12} className="sort-icon-inactive" />;
    };

    // Format cell value
    const formatValue = (value: unknown, colName: string): string => {
        if (value === null || value === undefined) return '—';
        if (typeof value === 'number') {
            // Format large numbers
            if (colName.includes('baseline') || colName.includes('retrofit') || colName.includes('saving')) {
                return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
            }
            return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
        }
        return String(value);
    };

    return (
        <div className="results-table-container">
            <table className="results-table">
                <thead>
                    <tr>
                        {results.columns.map((col, i) => (
                            <th key={col} onClick={() => handleSort(i)}>
                                <span className="th-content">
                                    <span className="th-label">{col}</span>
                                    {getSortIcon(i)}
                                </span>
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {sortedData.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                            {row.map((cell, colIndex) => (
                                <td key={colIndex} className={typeof cell === 'number' ? 'numeric' : ''}>
                                    {formatValue(cell, results.columns[colIndex])}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
