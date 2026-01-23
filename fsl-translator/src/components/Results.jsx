import React, { useState } from 'react';
import ImageSlider from './ImageSlider.jsx';
import BackButton from './BackButton.jsx';

const SERVER_BASE = 'http://localhost:8000';

export default function Results({ urls = [], result = null, loading = false, error = null, onBack }) {
    const [clearing, setClearing] = useState(false);
    const [clearError, setClearError] = useState(null);

    const handleBack = async () => {
        setClearError(null);
        setClearing(true);
        try {
            const res = await fetch(`${SERVER_BASE}/api/clear`, { method: 'POST' });
            let data = null;
            try { data = await res.json(); } catch (_ERR) { /* ignore non-json */ }

            if (!res.ok || (data && data.ok === false)) {
                const msg = data?.message || data?.error || `Clear failed: ${res.status}`;
                throw new Error(msg);
            }

            if (typeof onBack === 'function') onBack();
        } catch (err) {
            console.error('Error clearing uploads:', err);
            setClearError(err.message || 'Failed to clear uploads');
        } finally {
            setClearing(false);
        }
    };

    const renderMeta = (meta) => {
        if (!meta) return <div className="text-sm text-gray-600">No metadata</div>;
        return (
            <div className="text-left">
                {Object.entries(meta).map(([k, v]) => (
                    <div key={k} className="mb-2">
                        <div className="text-xs text-gray-500">{k}</div>
                        <div className="font-medium text-gray-800 break-words">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</div>
                    </div>
                ))}
            </div>
        );
    };

    // ⬅️ Get top3 from either top3 or top_3
    const top3Results = result?.top3 || result?.top_3;

    return (
        <div className="max-w-5xl mx-auto bg-white rounded shadow p-4">
            <div className="flex gap-6">
                <div className="w-1/2">
                    <h4 className="text-lg font-semibold mb-3">Preview</h4>
                    <ImageSlider images={urls} />
                </div>

                <div className="w-1/2 text-left">
                    <h4 className="text-lg font-semibold mb-3">Classification Results</h4>

                    {loading && <div className="text-gray-600 mb-2">Loading results...</div>}
                    {error && <div className="text-red-600 mb-2">{error}</div>}
                    {result ? (
                        <div className="space-y-4">
                            <div>
                                <div className="text-xs text-gray-500 mb-2">Top 3 Predictions</div>
                                {top3Results && top3Results.length > 0 ? (
                                    <ol className="list-decimal list-inside space-y-3">
                                        {top3Results.map((item, idx) => (
                                            <li key={idx} className={idx === 0 ? 'text-xl font-bold text-gray-900' : 'text-lg text-gray-700'}>
                                                {item.label}
                                                <span className="text-sm text-gray-600 ml-2">
                                                    {(item.confidence * 100).toFixed(1)}%
                                                </span>
                                            </li>
                                        ))}
                                    </ol>
                                ) : (
                                    <div className="text-xl font-bold text-gray-900">
                                        {result.prediction ?? '—'}
                                        <span className="text-sm text-gray-600 ml-2">
                                            {result.confidence ? `${(result.confidence * 100).toFixed(1)}%` : '—'}
                                        </span>
                                    </div>
                                )}
                            </div>

                        </div>
                    ) : (
                        <div className="text-gray-600">No results yet</div>
                    )}

                    {clearError && <div className="text-red-600 mt-3">{clearError}</div>}

                    <div className="mt-6 flex gap-3">
                        <BackButton onClick={handleBack} disabled={clearing} />
                    </div>
                </div>
            </div>
        </div>
    );
}