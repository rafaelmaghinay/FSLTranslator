import React, { useState, useRef } from 'react';

const SERVER_BASE = 'http://localhost:8000';

export default function Upload({ onUploadSuccess }) {
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);

    const fileRefs = {
        image: useRef(null),
        video: useRef(null),
    };

    const handleUpload = async (type) => {
        const input = fileRefs[type].current;
        if (!input?.files?.[0]) {
            setError('No file selected');
            return;
        }

        const file = input.files[0];
        const formData = new FormData();
        formData.append('file', file);

        setUploading(true);
        setError(null);

        try {
            const res = await fetch(`${SERVER_BASE}/api/upload`, {
                method: 'POST',
                body: formData,
            });

            const text = await res.text();
            let data;
            try {
                data = text ? JSON.parse(text) : {};
            } catch {
                throw new Error(`Invalid JSON: ${text}`);
            }

            if (!res.ok) {
                throw new Error(data?.error || data?.detail || `Upload failed: ${res.status}`);
            }

            if (typeof onUploadSuccess === 'function') {
                onUploadSuccess(data);
            }
        } catch (err) {
            setError(err.message || 'Upload failed');
            console.error('Upload error:', err);
        } finally {
            setUploading(false);
        }
    };

    const CardButton = ({ type, accept, label, icon }) => (
        <div className="bg-white rounded-lg shadow-md p-8 hover:shadow-lg transition-shadow flex flex-col items-center justify-center">
            <div className="text-6xl mb-4 flex items-center justify-center">{icon}</div>
            <h3 className="text-xl font-semibold mb-4 text-center">{label}</h3>
            <input
                ref={fileRefs[type]}
                type="file"
                accept={accept}
                className="hidden"
                onChange={() => handleUpload(type)}
            />
            <button
                onClick={() => fileRefs[type].current?.click()}
                disabled={uploading}
                className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded disabled:bg-gray-400 flex items-center justify-center"
            >
                {uploading ? 'Uploading...' : `Choose ${label}`}
            </button>
        </div>
    );

    return (
        <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold text-center mb-8">Upload Media</h2>

            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6 text-center">
                    {error}
                </div>
            )}

            <div className="grid md:grid-cols-2 gap-6">
                <CardButton
                    type="image"
                    accept="image/*"
                    label="Image"
                    icon="🖼️"
                />
                <CardButton
                    type="video"
                    accept="video/*"
                    label="Video"
                    icon="🎥"
                />
            </div>
        </div>
    );
}
