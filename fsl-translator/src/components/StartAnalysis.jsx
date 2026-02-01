// javascript
import React, { useState } from 'react';
import ImageSlider from './ImageSlider.jsx';
import BackButton from './BackButton.jsx';
import { SERVER_BASE } from '../config.js';


export default function StartAnalysis({ urls = [], paths = [], onResults, onBack }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleBack = async () => {
        setError(null);
        try {
            await fetch(`${SERVER_BASE}/api/clear`, { method: 'POST' });
        } catch (err) {
            console.error(err);
        }
        if (onBack) onBack();
    };

    const handleStart = async () => {
        setError(null);

        if (!Array.isArray(paths) || paths.length === 0) {
            setError('No cropped image paths available to classify.');
            return;
        }

        const payload = { cropped_images: paths.map(String) };

        setLoading(true);
        try {
            const res = await fetch(`${SERVER_BASE}/api/classify`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const text = await res.text();
            let data;
            try {
                data = text ? JSON.parse(text) : {};
            } catch {
                throw new Error(`Invalid JSON response: ${text}`);
            }

            if (!res.ok) {
                const errMsg = data?.error || data?.detail || JSON.stringify(data) || `HTTP ${res.status}`;
                throw new Error(errMsg);
            }

            if (typeof onResults === 'function') {
                onResults(data);
            }
        } catch (err) {
            setError(err.message || 'Analysis failed');
            console.error('Classification error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-md mx-auto text-center">
            <h3 className="text-2xl font-semibold mb-4">Ready for Analysis</h3>

            <div className="mb-4">
                <ImageSlider images={urls} />
            </div>

            {error && <p className="text-red-600 mb-2">{error}</p>}

            <div className="flex gap-3 justify-center mb-4">
                <BackButton onClick={handleBack} disabled={loading} />
                <button
                    onClick={handleStart}
                    disabled={loading}
                    className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-semibold rounded"
                >
                    {loading ? 'Classifying...' : 'Start Classification'}
                </button>
            </div>
        </div>
    );
}
