// TwinQL Console - Query editor and results panel (Enhanced)
import { useCallback, useState, useRef } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import Editor from '@monaco-editor/react';
import {
    Play, Loader2, Table2, BarChart3, Code2, History, Download, X,
    FileCode, ChevronDown, Sparkles
} from 'lucide-react';
import { useQueryStore, useUIStore } from '../../store';
import { api } from '../../services/api';
import { queryTemplates } from '../../utils/queryTemplates';
import ResultsTable from './ResultsTable';
import ResultsChart from './ResultsChart';
import './Console.css';

export default function Console() {
    const {
        currentQuery,
        setQuery,
        isExecuting,
        setExecuting,
        results,
        setResults,
        error,
        setError,
        activeTab,
        setActiveTab,
        history,
        addToHistory,
    } = useQueryStore();

    const { addNotification } = useUIStore();

    const [showHistory, setShowHistory] = useState(false);
    const [showTemplates, setShowTemplates] = useState(false);
    const templatesRef = useRef<HTMLDivElement>(null);

    // Execute query
    const handleExecute = useCallback(async () => {
        if (!currentQuery.trim() || isExecuting) return;

        setExecuting(true);
        setError(null);

        try {
            const result = await api.executeQuery(currentQuery);
            setResults(result);
            addNotification('success', `Query executed in ${result.execution_time_ms}ms`);

            addToHistory({
                id: Date.now().toString(),
                name: currentQuery.split('\n')[0].replace('--', '').trim() || 'Query',
                query: currentQuery,
                created_at: new Date().toISOString(),
                is_pinned: false,
            });
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Unknown error';
            setError({ message });
            addNotification('error', `Query failed: ${message}`);
        } finally {
            setExecuting(false);
        }
    }, [currentQuery, isExecuting, setExecuting, setResults, setError, addToHistory, addNotification]);

    // Keyboard shortcut
    const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            handleExecute();
        }
    }, [handleExecute]);

    // Apply template
    const handleTemplateSelect = (query: string) => {
        setQuery(query);
        setShowTemplates(false);
        addNotification('info', 'Template applied');
    };

    // Export results
    const handleExport = useCallback(() => {
        if (!results) return;

        const csv = [
            results.columns.join(','),
            ...results.data.map(row => row.join(',')),
        ].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `twinql_results_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        addNotification('success', 'Results exported to CSV');
    }, [results, addNotification]);

    // Calculate summary stats
    const summaryStats = results ? (() => {
        const baselineIdx = results.columns.findIndex(c => c.toLowerCase().includes('baseline'));
        const retrofitIdx = results.columns.findIndex(c => c.toLowerCase().includes('retrofit'));

        if (baselineIdx >= 0 && retrofitIdx >= 0) {
            const totalBaseline = results.data.reduce((sum, row) => sum + (Number(row[baselineIdx]) || 0), 0);
            const totalRetrofit = results.data.reduce((sum, row) => sum + (Number(row[retrofitIdx]) || 0), 0);
            const totalSaving = totalBaseline - totalRetrofit;
            const savingPct = totalBaseline > 0 ? (totalSaving / totalBaseline) * 100 : 0;

            return { totalBaseline, totalRetrofit, totalSaving, savingPct };
        }
        return null;
    })() : null;

    return (
        <div className="console" onKeyDown={handleKeyDown}>
            <PanelGroup direction="vertical" className="console-panels">
                {/* Query Editor Panel */}
                <Panel defaultSize={35} minSize={15}>
                    <div className="console-editor">
                        <div className="console-editor-header">
                            <span className="console-title">
                                <FileCode size={16} />
                                TwinQL Query
                            </span>
                            <div className="console-actions">
                                {/* Templates Dropdown */}
                                <div className="templates-dropdown" ref={templatesRef}>
                                    <button
                                        className="console-btn"
                                        onClick={() => setShowTemplates(!showTemplates)}
                                    >
                                        <Sparkles size={14} />
                                        <span>Templates</span>
                                        <ChevronDown size={14} className={showTemplates ? 'rotate' : ''} />
                                    </button>
                                    {showTemplates && (
                                        <div className="templates-menu">
                                            {queryTemplates.map(template => (
                                                <button
                                                    key={template.id}
                                                    className="template-item"
                                                    onClick={() => handleTemplateSelect(template.query)}
                                                >
                                                    <span className="template-name">{template.name}</span>
                                                    <span className="template-desc">{template.description}</span>
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                <button
                                    className="console-btn icon-btn"
                                    onClick={() => setShowHistory(!showHistory)}
                                    title="Query History"
                                >
                                    <History size={16} />
                                </button>
                                <button
                                    className="console-btn primary"
                                    onClick={handleExecute}
                                    disabled={isExecuting || !currentQuery.trim()}
                                    title="Run Query (Ctrl+Enter)"
                                >
                                    {isExecuting ? (
                                        <Loader2 size={16} className="spin" />
                                    ) : (
                                        <Play size={16} />
                                    )}
                                    <span>{isExecuting ? 'Running...' : 'Run'}</span>
                                </button>
                            </div>
                        </div>

                        <div className="console-editor-content">
                            <Editor
                                height="100%"
                                defaultLanguage="sql"
                                value={currentQuery}
                                onChange={(value) => setQuery(value || '')}
                                theme="vs-dark"
                                options={{
                                    minimap: { enabled: false },
                                    fontSize: 13,
                                    fontFamily: 'var(--font-mono)',
                                    lineNumbers: 'on',
                                    scrollBeyondLastLine: false,
                                    automaticLayout: true,
                                    wordWrap: 'on',
                                    padding: { top: 12, bottom: 12 },
                                }}
                            />
                        </div>

                        <div className="console-editor-footer">
                            <span className="console-hint">
                                <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to run • <kbd>Ctrl</kbd>+<kbd>Space</kbd> for suggestions
                            </span>
                        </div>
                    </div>
                </Panel>

                {/* Resize Handle */}
                <PanelResizeHandle className="console-resize-handle" />

                {/* Results Panel */}
                <Panel defaultSize={65} minSize={20}>
                    <div className="console-results">
                        <div className="console-results-header">
                            <div className="console-tabs">
                                <button
                                    className={`console-tab ${activeTab === 'table' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('table')}
                                >
                                    <Table2 size={14} />
                                    <span>Table</span>
                                </button>
                                <button
                                    className={`console-tab ${activeTab === 'chart' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('chart')}
                                >
                                    <BarChart3 size={14} />
                                    <span>Chart</span>
                                </button>
                                <button
                                    className={`console-tab ${activeTab === 'json' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('json')}
                                >
                                    <Code2 size={14} />
                                    <span>JSON</span>
                                </button>
                            </div>

                            {results && (
                                <div className="console-results-info">
                                    <span className="results-count">{results.data.length} rows</span>
                                    <span className="results-time">{results.execution_time_ms}ms</span>
                                    <button className="console-btn icon-btn" onClick={handleExport} title="Export CSV">
                                        <Download size={14} />
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Summary Stats Banner */}
                        {summaryStats && activeTab === 'chart' && (
                            <div className="results-summary">
                                <div className="summary-stat">
                                    <span className="summary-label">Total Baseline</span>
                                    <span className="summary-value">{summaryStats.totalBaseline.toLocaleString()} kWh</span>
                                </div>
                                <div className="summary-stat">
                                    <span className="summary-label">Total Retrofit</span>
                                    <span className="summary-value">{summaryStats.totalRetrofit.toLocaleString()} kWh</span>
                                </div>
                                <div className="summary-stat highlight">
                                    <span className="summary-label">Total Savings</span>
                                    <span className="summary-value">
                                        {summaryStats.totalSaving.toLocaleString()} kWh
                                        <span className="summary-pct">({summaryStats.savingPct.toFixed(1)}%)</span>
                                    </span>
                                </div>
                            </div>
                        )}

                        <div className="console-results-content">
                            {error && (
                                <div className="console-error">
                                    <X size={16} />
                                    <span>{error.message}</span>
                                </div>
                            )}

                            {!error && !results && !isExecuting && (
                                <div className="console-empty">
                                    <FileCode size={32} className="empty-icon" />
                                    <span className="empty-title">No Results Yet</span>
                                    <span className="empty-desc">Write a query and click Run, or select a template to get started</span>
                                </div>
                            )}

                            {isExecuting && (
                                <div className="console-loading">
                                    <Loader2 size={32} className="spin" />
                                    <span>Executing query...</span>
                                </div>
                            )}

                            {results && activeTab === 'table' && <ResultsTable results={results} />}
                            {results && activeTab === 'chart' && <ResultsChart results={results} />}
                            {results && activeTab === 'json' && (
                                <div className="console-json">
                                    <pre>{JSON.stringify(results, null, 2)}</pre>
                                </div>
                            )}
                        </div>

                        {results && (
                            <div className="console-results-footer">
                                <details className="console-explain">
                                    <summary>Execution Plan</summary>
                                    <pre>{results.explain}</pre>
                                </details>
                            </div>
                        )}
                    </div>
                </Panel>
            </PanelGroup>

            {/* Query History Sidebar */}
            {showHistory && (
                <div className="console-history">
                    <div className="console-history-header">
                        <span>Query History</span>
                        <button className="icon-btn" onClick={() => setShowHistory(false)}>
                            <X size={16} />
                        </button>
                    </div>
                    <div className="console-history-list">
                        {history.length === 0 ? (
                            <div className="console-history-empty">No queries yet</div>
                        ) : (
                            history.map((q) => (
                                <button
                                    key={q.id}
                                    className="console-history-item"
                                    onClick={() => {
                                        setQuery(q.query);
                                        setShowHistory(false);
                                    }}
                                >
                                    <span className="history-name">{q.name}</span>
                                    <span className="history-preview">
                                        {q.query.split('\n').slice(0, 2).join(' ').substring(0, 50)}...
                                    </span>
                                    <span className="history-time">
                                        {new Date(q.created_at).toLocaleTimeString()}
                                    </span>
                                </button>
                            ))
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
