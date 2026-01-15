// File: `src/components/Upload.jsx`
import React, { useRef, useState } from 'react';

const SERVER_BASE = 'http://localhost:8000';

export default function Upload({ onUploaded, onStartLive }) {
    const fileRefs = {
        video: useRef(null),
        image: useRef(null),
        multiple: useRef(null)
    };
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);

    const cards = [
        { title: 'Upload a Video', icon:"🎥", desc:"Upload", key: 'video', accept: 'video/*', multiple: false },
        { title: 'Upload an Image', icon:"🖼️", desc:"Upload",key: 'image', accept: 'image/*', multiple: false },
        { title: '20 Image Sequence', icon:"📁", desc:"Upload",key : 'multiple', accept: 'image/*', multiple: true },
        { title: 'Start Live Camera', icon:"📷", desc:"Start", key: 'live', accept: '', multiple: false }
    ];

    const toUrl = (p, typeHint) => {
        if (!p) return null;
        if (p.startsWith('http') || p.startsWith('data:')) return p;
        if (p.startsWith('/')) return `${SERVER_BASE}${p}`;
        if (p.includes('uploads')) return `${SERVER_BASE}/${p.replace(/^\/+/, '')}`;
        if (typeHint === 'sequence' || typeHint === 'sequences') return `${SERVER_BASE}/uploads/sequences/${p}`;
        if (typeHint === 'image' || typeHint === 'images') return `${SERVER_BASE}/uploads/images/${p}`;
        return `${SERVER_BASE}/uploads/${p}`;
    };

    const openFile = (key) => {
        if (key === 'live') {
            setError(null);
            if (typeof onStartLive === 'function') onStartLive();
            return;
        }
        const ref = fileRefs[key];
        if (ref && ref.current) ref.current.click();
    };

    const handleChange = (e) => {
        const el = e.target;
        if (!el.files || el.files.length === 0) return;
        if (el.multiple) {
            uploadSequence(Array.from(el.files));
        } else {
            const f = el.files[0];
            if (el.id === 'videoInput') uploadVideo(f); else uploadImage(f);
        }
        el.value = '';
    };

    async function uploadImage(file) {
        setUploading(true);
        setError(null);
        try {
            if (!file.type || !file.type.startsWith('image/')) throw new Error('Only image files allowed');
            const fd = new FormData();
            fd.append('image', file);
            const res = await fetch(`${SERVER_BASE}/api/upload/image`, { method: 'POST', body: fd });
            if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
            const json = await res.json();
            if (!json.ok) throw new Error(json.error || 'Upload failed');
            const path = json.cropped_image || json.cropped_image_path || json.saved_as;
            const url = toUrl(path, 'images');
            onUploaded({ urls: [url], paths: [path], type: 'image' });
        } catch (err) {
            setError(err.message || 'Upload error');
            console.error(err);
        } finally {
            setUploading(false);
        }
    }

    async function uploadSequence(files) {
        setUploading(true);
        setError(null);
        try {
            if (!Array.isArray(files)) files = Array.from(files);
            if (files.length !== 20) throw new Error('Please select exactly 20 images');
            for (const f of files) {
                if (!f.type || !f.type.startsWith('image/')) throw new Error('All files must be images');
            }
            const fd = new FormData();
            files.forEach((f) => fd.append('images', f));
            const res = await fetch(`${SERVER_BASE}/api/upload/sequence`, { method: 'POST', body: fd });
            if (!res.ok) {
                const text = await res.text();
                throw new Error(`Upload failed: ${res.status} ${text}`);
            }
            const json = await res.json();
            if (!json.ok) throw new Error(json.error || 'Sequence upload failed');
            const raw = Array.isArray(json.cropped_images) ? json.cropped_images : [];
            const urls = raw.map((p) => toUrl(p, 'sequences'));
            onUploaded({ urls, paths: raw, type: 'sequence' });
        } catch (err) {
            setError(err.message || 'Upload error');
            console.error(err);
        } finally {
            setUploading(false);
        }
    }

    async function uploadVideo(file) {
        setUploading(true);
        setError(null);
        try {
            if (!file.type || !file.type.startsWith('video/')) throw new Error('Only video files allowed');
            const fd = new FormData();
            fd.append('video', file);
            const res = await fetch(`${SERVER_BASE}/api/upload/video`, { method: 'POST', body: fd });
            if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
            const json = await res.json();
            if (!json.ok) throw new Error(json.error || 'Video upload failed');
            const raw = Array.isArray(json.cropped_images) ? json.cropped_images : [];
            const urls = raw.map((p) => toUrl(p, 'sequences'));
            onUploaded({ urls, paths: raw, type: 'video' });
        } catch (err) {
            setError(err.message || 'Upload error');
            console.error(err);
        } finally {
            setUploading(false);
        }
    }

    return (
        <div>
            <div className="mt-12 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto">
                {cards.map((c) => (
                    <div key={c.key} className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow">
                        <div className="text-3xl mb-3">{c.icon}</div>
                        <h3 className="text-xl font-semibold text-gray-900">{c.title}</h3>
                        <button onClick={() => openFile(c.key)} className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-1 px-2 rounded transition-colors text-sm">
                            {c.desc}
                        </button>
                    </div>
                ))}
            </div>

            <input id="videoInput" ref={fileRefs.video} type="file" accept="video/*" hidden onChange={handleChange} />
            <input id="imageInput" ref={fileRefs.image} type="file" accept="image/*" hidden onChange={handleChange} />
            <input id="multipleInput" ref={fileRefs.multiple} type="file" accept="image/*" multiple hidden onChange={handleChange} />

            {uploading && <p className="text-center mt-4">Uploading...</p>}
            {error && <p className="text-red-600 text-center mt-2">{error}</p>}
        </div>
    );
}
