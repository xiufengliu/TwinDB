// Data Upload Dialog - Upload CSV time-series data to a twin
import { useState, useCallback, useRef } from 'react';
import { X, Upload, FileSpreadsheet, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import { useUIStore } from '../../store';
import { api } from '../../services/api';
import './DataUpload.css';

interface DataUploadProps {
    twinId: string;
    onClose: () => void;
}

interface ParsedData {
    headers: string[];
    rows: string[][];
    preview: string[][];
}

export default function DataUpload({ twinId, onClose }: DataUploadProps) {
    const { addNotification } = useUIStore();

    const [file, setFile] = useState<File | null>(null);
    const [parsedData, setParsedData] = useState<ParsedData | null>(null);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Column mapping
    const [tsColumn, setTsColumn] = useState<string>('');
    const [valueColumn, setValueColumn] = useState<string>('');
    const [metricName, setMetricName] = useState<string>('heat_load');

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Parse CSV file
    const parseCSV = useCallback((text: string): ParsedData => {
        const lines = text.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
        const rows = lines.slice(1).map(line =>
            line.split(',').map(cell => cell.trim().replace(/"/g, ''))
        );
        const preview = rows.slice(0, 5);
        return { headers, rows, preview };
    }, []);

    // Handle file selection
    const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const selected = e.target.files?.[0];
        if (!selected) return;

        if (!selected.name.endsWith('.csv')) {
            setError('Please select a CSV file');
            return;
        }

        setFile(selected);
        setError(null);

        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const text = event.target?.result as string;
                const data = parseCSV(text);
                setParsedData(data);

                // Auto-detect columns
                const tsIdx = data.headers.findIndex(h =>
                    h.toLowerCase().includes('time') ||
                    h.toLowerCase().includes('date') ||
                    h.toLowerCase() === 'ts'
                );
                const valueIdx = data.headers.findIndex(h =>
                    h.toLowerCase().includes('value') ||
                    h.toLowerCase().includes('load') ||
                    h.toLowerCase().includes('energy')
                );

                if (tsIdx >= 0) setTsColumn(data.headers[tsIdx]);
                if (valueIdx >= 0) setValueColumn(data.headers[valueIdx]);
            } catch (err) {
                setError('Failed to parse CSV file');
            }
        };
        reader.readAsText(selected);
    }, [parseCSV]);

    // Handle upload
    const handleUpload = useCallback(async () => {
        if (!parsedData || !tsColumn || !valueColumn) {
            setError('Please select timestamp and value columns');
            return;
        }

        setIsUploading(true);
        setError(null);

        try {
            const tsIdx = parsedData.headers.indexOf(tsColumn);
            const valueIdx = parsedData.headers.indexOf(valueColumn);

            const data = parsedData.rows.map(row => ({
                ts: row[tsIdx],
                value: parseFloat(row[valueIdx]) || 0,
            }));

            const result = await api.uploadTimeSeries({
                tid: twinId,
                metric: metricName,
                data,
            });

            addNotification('success', `Uploaded ${result.count} data points to ${twinId}`);
            onClose();
        } catch (err) {
            setError('Failed to upload data');
            addNotification('error', 'Upload failed');
        } finally {
            setIsUploading(false);
        }
    }, [parsedData, tsColumn, valueColumn, metricName, twinId, onClose, addNotification]);

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal data-upload-modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>Upload Time-Series Data</h3>
                    <button className="icon-btn-sm" onClick={onClose}>
                        <X size={18} />
                    </button>
                </div>

                <div className="modal-body">
                    <div className="upload-target-info">
                        <span className="label">Target Twin:</span>
                        <span className="value">{twinId}</span>
                    </div>

                    {/* File Selection */}
                    <div
                        className={`upload-dropzone ${file ? 'has-file' : ''}`}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".csv"
                            onChange={handleFileChange}
                            hidden
                        />
                        {file ? (
                            <div className="file-info">
                                <FileSpreadsheet size={24} />
                                <span className="file-name">{file.name}</span>
                                <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
                            </div>
                        ) : (
                            <div className="upload-placeholder">
                                <Upload size={32} />
                                <span>Click to select CSV file</span>
                                <span className="hint">or drag and drop</span>
                            </div>
                        )}
                    </div>

                    {/* Error Message */}
                    {error && (
                        <div className="upload-error">
                            <AlertCircle size={16} />
                            <span>{error}</span>
                        </div>
                    )}

                    {/* Column Mapping */}
                    {parsedData && (
                        <div className="column-mapping">
                            <h4>Map Columns</h4>
                            <div className="mapping-row">
                                <div className="mapping-field">
                                    <label>Timestamp Column</label>
                                    <select value={tsColumn} onChange={e => setTsColumn(e.target.value)}>
                                        <option value="">Select...</option>
                                        {parsedData.headers.map(h => (
                                            <option key={h} value={h}>{h}</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="mapping-field">
                                    <label>Value Column</label>
                                    <select value={valueColumn} onChange={e => setValueColumn(e.target.value)}>
                                        <option value="">Select...</option>
                                        {parsedData.headers.map(h => (
                                            <option key={h} value={h}>{h}</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="mapping-field">
                                    <label>Metric Name</label>
                                    <input
                                        type="text"
                                        value={metricName}
                                        onChange={e => setMetricName(e.target.value)}
                                        placeholder="e.g., heat_load"
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Data Preview */}
                    {parsedData && (
                        <div className="data-preview">
                            <h4>Preview ({parsedData.rows.length} rows total)</h4>
                            <div className="preview-table-wrapper">
                                <table className="preview-table">
                                    <thead>
                                        <tr>
                                            {parsedData.headers.map((h, i) => (
                                                <th key={i} className={
                                                    h === tsColumn ? 'selected-ts' :
                                                        h === valueColumn ? 'selected-value' : ''
                                                }>{h}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {parsedData.preview.map((row, i) => (
                                            <tr key={i}>
                                                {row.map((cell, j) => (
                                                    <td key={j} className={
                                                        parsedData.headers[j] === tsColumn ? 'selected-ts' :
                                                            parsedData.headers[j] === valueColumn ? 'selected-value' : ''
                                                    }>{cell}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>

                <div className="modal-footer">
                    <button className="btn-secondary" onClick={onClose}>
                        Cancel
                    </button>
                    <button
                        className="btn-primary"
                        onClick={handleUpload}
                        disabled={!parsedData || !tsColumn || !valueColumn || isUploading}
                    >
                        {isUploading ? (
                            <>
                                <Loader2 size={16} className="spin" />
                                <span>Uploading...</span>
                            </>
                        ) : (
                            <>
                                <CheckCircle size={16} />
                                <span>Upload Data</span>
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}
