import React, { useState } from 'react';
import ImageSlider from './ImageSlider.jsx';
import BackButton from './BackButton.jsx';
import { SERVER_BASE } from '../config.js';


// Display classification results with confidence scores and media preview
export default function Results({ urls = [], result = null, loading = false, error = null, onBack }) {
    const [clearing, setClearing] = useState(false);
    const [clearError, setClearError] = useState(null);
    const [showAll, setShowAll] = useState(false);

    // Handle back navigation and cleanup uploads from server
    const handleBack = async () => {
        setClearError(null);
        setClearing(true);
        try {
            const res = await fetch(`${SERVER_BASE}/api/clear`, { method: 'POST' });
            let data = null;
            try { data = await res.json(); } catch { /* ignore non-json */ }

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
                                {result.all_predictions && result.all_predictions.length > 0 ? (
                                    <>
                                        <table className="w-full border-collapse">
                                            <thead>
                                            <tr className="border-b border-gray-300">
                                                <th className="text-left py-2 px-3 font-semibold text-gray-700 w-12">No.</th>
                                                <th className="text-left py-2 px-3 font-semibold text-gray-700">Prediction</th>
                                                <th className="text-right py-2 px-3 font-semibold text-gray-700">Confidence Level</th>
                                            </tr>
                                            </thead>
                                            <tbody>
                                            {result.all_predictions.slice(0, 3).map((item, idx) => (
                                                <tr key={idx} className={`border-b border-gray-200 ${idx === 0 ? 'bg-blue-50' : ''}`}>
                                                    <td className={`py-2 px-3 ${idx === 0 ? 'text-lg font-bold text-gray-900' : 'text-base text-gray-700'}`}>
                                                        {idx + 1}
                                                    </td>
                                                    <td className={`py-2 px-3 ${idx === 0 ? 'text-lg font-bold text-gray-900' : 'text-base text-gray-700'}`}>
                                                        {item.label}
                                                    </td>
                                                    <td className={`py-2 px-3 text-right ${idx === 0 ? 'text-lg font-semibold text-gray-900' : 'text-base text-gray-600'}`}>
                                                        {(item.confidence * 100).toFixed(1)}%
                                                    </td>
                                                </tr>
                                            ))}
                                            </tbody>
                                        </table>



                                        {/* Show All Predictions Dropdown */}
                                        {result.all_predictions.length > 3 && (
                                            <details className="mt-3" open={showAll} onToggle={(e) => setShowAll(e.target.open)}>
                                                <summary className="cursor-pointer text-sm font-medium text-blue-600 hover:text-blue-800">
                                                    {showAll ? 'Show less' : `Show remaining ${result.all_predictions.length - 3} predictions`}
                                                </summary>
                                                <table className="w-full border-collapse mt-2">
                                                    <tbody>
                                                    {result.all_predictions.slice(3).map((item, idx) => (
                                                        <tr key={idx} className="border-b border-gray-100">
                                                            <td className="py-1.5 px-3 text-sm text-gray-700 w-12">
                                                                {idx + 4}
                                                            </td>
                                                            <td className="py-1.5 px-3 text-sm text-gray-700">
                                                                {item.label}
                                                            </td>
                                                            <td className="py-1.5 px-3 text-right text-sm text-gray-600">
                                                                {(item.confidence * 100).toFixed(2)}%
                                                            </td>
                                                        </tr>
                                                    ))}
                                                    </tbody>
                                                </table>
                                            </details>
                                        )}
                                    </>
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