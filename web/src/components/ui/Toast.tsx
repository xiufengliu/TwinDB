// Toast Notification System
import { useEffect } from 'react';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react';
import { useUIStore } from '../../store';
import './Toast.css';

const icons = {
    success: CheckCircle,
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info,
};

export default function ToastContainer() {
    const { notifications, removeNotification } = useUIStore();

    return (
        <div className="toast-container">
            {notifications.map((notification) => (
                <Toast
                    key={notification.id}
                    notification={notification}
                    onClose={() => removeNotification(notification.id)}
                />
            ))}
        </div>
    );
}

interface ToastProps {
    notification: { id: string; type: 'info' | 'success' | 'warning' | 'error'; message: string };
    onClose: () => void;
}

function Toast({ notification, onClose }: ToastProps) {
    const Icon = icons[notification.type];

    useEffect(() => {
        const timer = setTimeout(onClose, 4000);
        return () => clearTimeout(timer);
    }, [onClose]);

    return (
        <div className={`toast toast-${notification.type}`}>
            <Icon size={18} className="toast-icon" />
            <span className="toast-message">{notification.message}</span>
            <button className="toast-close" onClick={onClose}>
                <X size={14} />
            </button>
        </div>
    );
}
