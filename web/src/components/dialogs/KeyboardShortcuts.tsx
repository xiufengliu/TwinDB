// Keyboard Shortcuts Help Panel
import { X, Keyboard } from 'lucide-react';
import './KeyboardShortcuts.css';

interface KeyboardShortcutsProps {
    onClose: () => void;
}

const shortcuts = [
    {
        category: 'Query Editor', items: [
            { keys: ['Ctrl', 'Enter'], action: 'Execute query' },
            { keys: ['Ctrl', 'Space'], action: 'Show suggestions' },
            { keys: ['Ctrl', 'S'], action: 'Format query' },
            { keys: ['Ctrl', '/'], action: 'Toggle comment' },
        ]
    },
    {
        category: 'Canvas', items: [
            { keys: ['Delete'], action: 'Remove selected node' },
            { keys: ['Ctrl', 'A'], action: 'Select all nodes' },
            { keys: ['Ctrl', 'D'], action: 'Duplicate selection' },
            { keys: ['Escape'], action: 'Deselect all' },
            { keys: ['+'], action: 'Zoom in' },
            { keys: ['-'], action: 'Zoom out' },
            { keys: ['0'], action: 'Fit view' },
        ]
    },
    {
        category: 'Navigation', items: [
            { keys: ['Ctrl', '1'], action: 'Workspace view' },
            { keys: ['Ctrl', '2'], action: 'Console view' },
            { keys: ['Ctrl', '3'], action: 'Split view' },
            { keys: ['Ctrl', 'B'], action: 'Toggle sidebar' },
        ]
    },
    {
        category: 'General', items: [
            { keys: ['Ctrl', 'N'], action: 'New twin' },
            { keys: ['Ctrl', 'O'], action: 'Load workspace' },
            { keys: ['Ctrl', 'E'], action: 'Export workspace' },
            { keys: ['?'], action: 'Show shortcuts' },
        ]
    },
];

export default function KeyboardShortcuts({ onClose }: KeyboardShortcutsProps) {
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal shortcuts-modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <div className="modal-title">
                        <Keyboard size={20} />
                        <h3>Keyboard Shortcuts</h3>
                    </div>
                    <button className="icon-btn-sm" onClick={onClose}>
                        <X size={18} />
                    </button>
                </div>

                <div className="modal-body shortcuts-body">
                    <div className="shortcuts-grid">
                        {shortcuts.map(category => (
                            <div key={category.category} className="shortcuts-category">
                                <h4>{category.category}</h4>
                                <div className="shortcuts-list">
                                    {category.items.map((shortcut, i) => (
                                        <div key={i} className="shortcut-item">
                                            <div className="shortcut-keys">
                                                {shortcut.keys.map((key, j) => (
                                                    <span key={j}>
                                                        <kbd>{key}</kbd>
                                                        {j < shortcut.keys.length - 1 && <span className="plus">+</span>}
                                                    </span>
                                                ))}
                                            </div>
                                            <span className="shortcut-action">{shortcut.action}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="modal-footer shortcuts-footer">
                    <span className="shortcuts-hint">
                        Press <kbd>?</kbd> anytime to show this panel
                    </span>
                </div>
            </div>
        </div>
    );
}
