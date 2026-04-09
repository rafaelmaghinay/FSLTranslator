import React, { useRef, useState, useEffect } from 'react';
import BackButton from './BackButton.jsx';
import ImageSlider from './ImageSlider.jsx';

import { SERVER_BASE, WS_URL } from '../config.js';

// Real-time camera feed with hand detection and frame capture for live sign language recognition
export default function LiveCamera({ onResults, onBack }) {
    const videoRef = useRef(null);
    const overlayRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const streamRef = useRef(null);
    const frameIntervalRef = useRef(null);
    const captureIntervalRef = useRef(null);


    const [streaming, setStreaming] = useState(false);
    const [recording, setRecording] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [sessionId, setSessionId] = useState(null);
    const [frameCount, setFrameCount] = useState(0);
    const [croppedUrls, setCroppedUrls] = useState([]);
    const [_result, setResult] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [classificationResults, setClassificationResults] = useState(null);


    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopStream();
        };
    }, []);

    useEffect(() => {
        const handleKeyPress = (event) => {
            // Check if spacebar is pressed and camera is streaming
            if (event.code === 'Space' && streaming && !loading) {
                event.preventDefault(); // Prevent page scroll
                handleCaptureToggle();
            }
        };

        window.addEventListener('keydown', handleKeyPress);

        return () => {
            window.removeEventListener('keydown', handleKeyPress);
        };
    }, [streaming, recording, loading]);

    // Start camera and WebSocket connection
    const startStream = async () => {
        try {
            setError('');

            // Request exactly 640x640 resolution
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: {ideal: 640},
                    height: {ideal: 640},
                    facingMode: 'user'
                }
            });

            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }

            // Connect WebSocket
            const ws = new WebSocket(WS_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('🔌 WebSocket connected');
                setStreaming(true);
                startSendingFrames();
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.error) {
                    console.error('Server error:', data.error);
                    return;
                }

                setResult(data);
                drawBoxes(data.boxes || []);
            };

            ws.onerror = (err) => {
                console.error('WebSocket error:', err);
                setError('Connection error');
            };

            ws.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                setStreaming(false);
            };

        } catch (err) {
            console.error('Failed to start camera:', err);
            setError('Camera access denied');
        }
    };

    // Stop camera and WebSocket
    const stopStream = () => {
        // Stop sending frames
        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current);
            frameIntervalRef.current = null;
        }

        if (captureIntervalRef.current) {
            clearInterval(captureIntervalRef.current);
            captureIntervalRef.current = null;
        }

        // Close WebSocket
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        // Stop video stream
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }

        setStreaming(false);
        setRecording(false);
        setFrameCount(0);
        clearOverlay();
    };

    // Send frames to WebSocket for detection
    const startSendingFrames = () => {
        frameIntervalRef.current = setInterval(() => {
            if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
            if (!videoRef.current || videoRef.current.paused) return;

            const canvas = canvasRef.current;
            const video = videoRef.current;

            // Resize to 640x640 before sending
            canvas.width = 640;
            canvas.height = 640;

            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, 640, 640);

            // Lower quality for faster transmission
            const frameBase64 = canvas.toDataURL('image/jpeg', 0.4);

            wsRef.current.send(JSON.stringify({
                frame: frameBase64
            }));
        }, 200);

    // Draw detection boxes on overlay
    const drawBoxes = (boxes) => {
        const overlay = overlayRef.current;
        const video = videoRef.current;

        if (!overlay || !video) return;

        // Match overlay to video dimensions
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;
        const displayWidth = video.offsetWidth;
        const displayHeight = video.offsetHeight;

        overlay.width = displayWidth;
        overlay.height = displayHeight;

        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        // Calculate scaling factors
        const scaleX = displayWidth / videoWidth;
        const scaleY = displayHeight / videoHeight;

        boxes.forEach(box => {
            // Scale coordinates to display size
            const x1 = box.x1 * scaleX;
            const y1 = box.y1 * scaleY;
            const x2 = box.x2 * scaleX;
            const y2 = box.y2 * scaleY;
            const width = x2 - x1;
            const height = y2 - y1;

            // Clamp to canvas bounds
            const clampedX1 = Math.max(0, Math.min(x1, displayWidth));
            const clampedY1 = Math.max(0, Math.min(y1, displayHeight));
            const clampedWidth = Math.min(width, displayWidth - clampedX1);
            const clampedHeight = Math.min(height, displayHeight - clampedY1);

            // Draw box
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 3;
            ctx.strokeRect(clampedX1, clampedY1, clampedWidth, clampedHeight);

            // Draw confidence
            ctx.fillStyle = '#00FF00';
            ctx.font = '16px Arial';
            ctx.fillText(`${(box.conf * 100).toFixed(1)}%`, clampedX1, Math.max(16, clampedY1 - 5));
        });
    };

    const clearOverlay = () => {
        const overlay = overlayRef.current;
        if (!overlay) return;
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    };

    // Toggle capture/stop
    const handleCaptureToggle = async () => {
        if (recording) {
            await handleStopCapture();
        } else {
            await handleStartCapture();
        }
    };

    // Start capturing frames
    const handleStartCapture = async () => {
        try {
            setLoading(true);
            setError('');
            setCroppedUrls([]);
            setFrameCount(0);

            // Start capture session
            const formData = new FormData();
            if (sessionId) formData.append('session_id', sessionId);

            const res = await fetch(`${SERVER_BASE}/api/live/capture`, {
                method: 'POST',
                body: formData
            });

            const data = await res.json();

            if (!data.ok) {
                throw new Error(data.error || 'Failed to start capture');
            }

            setSessionId(data.session_id);
            setRecording(true);

            // Start sending frames for capture
            startCapturingFrames(data.session_id);

        } catch (err) {
            console.error('Failed to start capture:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Send frames to API for capture
    const startCapturingFrames = (sid) => {
        let count = 0;

        captureIntervalRef.current = setInterval(async () => {
            if (!videoRef.current || videoRef.current.paused) return;

            const canvas = canvasRef.current;
            const video = videoRef.current;

            canvas.width = 640;
            canvas.height = 640;

            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, 640, 640);

            const frameBase64 = canvas.toDataURL('image/jpeg', 0.7);
            const timestamp = Date.now() / 1000; // Unix timestamp in seconds (float)

            try {
                const formData = new FormData();
                formData.append('session_id', sid);
                formData.append('frame', frameBase64);
                formData.append('timestamp', timestamp.toString());

                const res = await fetch(`${SERVER_BASE}/api/live/frame`, {
                    method: 'POST',
                    body: formData
                });

                const data = await res.json();

                if (data.ok) {
                    count++;
                    setFrameCount(data.hand_frames || count);
                }
            } catch (err) {
                console.error('Failed to send frame:', err);
            }
        }, 150);
    };


    const handleStopCapture = async () => {
        try {
            setLoading(true);
            setError('');

            // Stop sending capture frames
            if (captureIntervalRef.current) {
                clearInterval(captureIntervalRef.current);
                captureIntervalRef.current = null;
            }

            if (!sessionId) {
                setRecording(false); // Reset recording state even if no session
                throw new Error('No session ID');
            }

            const formData = new FormData();
            formData.append('session_id', sessionId);

            const res = await fetch(`${SERVER_BASE}/api/live/stop`, {
                method: 'POST',
                body: formData
            });

            const data = await res.json();

            if (!data.ok) {
                throw new Error(data.error || 'Failed to stop capture');
            }

            setCroppedUrls(data.cropped_images || []);
            setRecording(false);

            // Show message if no hands detected
            if (!data.cropped_images || data.cropped_images.length === 0) {
                setError('No hands detected. Please try again.');
            }

            // Pass results to parent
            if (onResults) {
                onResults(data.cropped_images);
            }

        } catch (err) {
            console.error('Failed to stop capture:', err);
            setError(err.message);
            setRecording(false); // Always reset recording state
        } finally {
            setSessionId(null); // Always reset session ID to allow new captures
            setLoading(false);
        }
    };

    const handleBack = async () => {
        stopStream();

        // Always clear session on backend
        try {
            await fetch(`${SERVER_BASE}/api/clear`, {
                method: 'POST'
            });
        } catch (err) {
            console.error('Failed to clear session:', err);
        }

        if (onBack) onBack();
    };

    const handleAnalyze = async () => {
        if (croppedUrls.length === 0) {
            setError('No frames to analyze');
            return;
        }

        try {
            setAnalyzing(true);
            setError('');

            console.log(`🔍 Analyzing ${croppedUrls.length} cropped frames...`);

            const res = await fetch(`${SERVER_BASE}/api/classify`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    cropped_images: croppedUrls
                })
            });

            const data = await res.json();

            if (!data.ok) {
                throw new Error(data.error || 'Classification failed');
            }

            console.log(`✅ Prediction: ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`);

            // Extract top 3 and all predictions
            const top3 = data.all_predictions?.slice(0, 3) || [];

            setClassificationResults({
                prediction: data.prediction,
                confidence: data.confidence,
                frames_used: data.frames_used,
                top3: top3,
                allPredictions: data.all_predictions || []
            });

        } catch (err) {
            console.error('Failed to analyze:', err);
            setError(err.message);
        } finally {
            setAnalyzing(false);
        }
    };

    const [showAll, setShowAll] = useState(false);

    return (
        <div className="max-w-3xl mx-auto bg-white rounded shadow p-4 text-center">
            <h3 className="text-2xl font-semibold mb-4">Live Camera</h3>

            {/* Responsive layout: stack on mobile, side-by-side on md+ */}
            <div className="mb-3 flex flex-col md:flex-row gap-4 items-start">
                <div className="w-full md:w-1/2 relative">
                    <video ref={videoRef} className="w-full bg-black rounded" playsInline muted/>
                    <canvas ref={overlayRef} className="absolute inset-0 pointer-events-none"/>
                    <canvas ref={canvasRef} style={{display: 'none'}}/>
                </div>

                <div className="w-full md:w-1/2">
                    <h4 className="text-sm font-semibold mb-2">Captured Frames</h4>
                    {croppedUrls.length ? (
                        <ImageSlider images={croppedUrls} className="max-h-72"/>
                    ) : (
                        <div className="text-gray-600">No frames yet</div>
                    )}
                </div>
            </div>

            {/* Responsive buttons: stack on mobile, row on sm+ */}
            <div className="flex flex-col sm:flex-row gap-3 justify-center mt-10 mb-2">
                {!streaming ? (
                    <button onClick={startStream}
                            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded">
                        Start Camera
                    </button>
                ) : (
                    <button onClick={stopStream}
                            className="px-4 py-2 bg-gray-300 hover:bg-gray-400 text-gray-800 font-semibold rounded">
                        Stop Camera
                    </button>
                )}

                <button
                    onClick={handleCaptureToggle}
                    disabled={!streaming || loading}
                    className={`px-4 py-2 ${recording ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'} text-white font-semibold rounded disabled:opacity-50`}
                >
                    {loading ? 'Processing...' : (recording ? 'Stop & Upload' : 'Capture Video')}
                </button>

                <button
                    onClick={handleAnalyze}
                    disabled={!croppedUrls.length || analyzing || loading}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white font-semibold rounded disabled:opacity-50"
                >
                    {analyzing ? 'Analyzing...' : 'Start Analysis'}
                </button>

                <BackButton onClick={handleBack}/>
            </div>

            {error && <div className="text-red-600 mb-2">{error}</div>}

            <div className="text-left mt-4">
                <div className="text-xs text-gray-500">Status</div>
                <div className="font-medium text-gray-800">
                    {streaming ? (recording ? `Recording... (${frameCount} frames)` : 'Camera active') : 'Camera stopped'}
                </div>
            </div>

            {classificationResults && (
                <div className="text-left mt-4 p-4 bg-green-50 rounded border border-green-200">
                    <div className="text-sm font-semibold text-green-800 mb-2">Classification Results</div>

                    <table className="w-full border-collapse">
                        <thead>
                        <tr className="border-b border-gray-300">
                            <th className="text-left py-2 px-3 font-semibold text-gray-700 w-12">#</th>
                            <th className="text-left py-2 px-3 font-semibold text-gray-700">Prediction</th>
                            <th className="text-right py-2 px-3 font-semibold text-gray-700">Confidence Level</th>
                        </tr>
                        </thead>
                        <tbody>
                        {classificationResults.allPredictions?.slice(0, 3).map((result, idx) => (
                            <tr key={idx} className={`border-b border-gray-200 ${idx === 0 ? 'bg-blue-50' : ''}`}>
                                <td className={`py-2 px-3 ${idx === 0 ? 'text-lg font-bold text-gray-900' : 'text-base text-gray-700'}`}>
                                    {idx + 1}
                                </td>
                                <td className={`py-2 px-3 ${idx === 0 ? 'text-lg font-bold text-gray-900' : 'text-base text-gray-700'}`}>
                                    {result.label}
                                </td>
                                <td className={`py-2 px-3 text-right ${idx === 0 ? 'text-lg font-semibold text-gray-900' : 'text-base text-gray-600'}`}>
                                    {(result.confidence * 100).toFixed(1)}%
                                </td>
                            </tr>
                        ))}
                        </tbody>
                    </table>



                    {/* Show All Predictions Dropdown */}
                    {classificationResults.allPredictions && classificationResults.allPredictions.length > 3 && (
                        <details className="mt-3" open={showAll} onToggle={(e) => setShowAll(e.target.open)}>
                            <summary className="cursor-pointer text-sm font-medium text-blue-600 hover:text-blue-800">
                                {showAll ? 'Show less' : `Show remaining ${classificationResults.allPredictions.length - 3} predictions`}
                            </summary>
                            <table className="w-full border-collapse mt-2">
                                <tbody>
                                {classificationResults.allPredictions.slice(3).map((result, idx) => (
                                    <tr key={idx} className="border-b border-gray-100">
                                        <td className="py-1.5 px-3 text-sm text-gray-700 w-12">
                                            {idx + 4}
                                        </td>
                                        <td className="py-1.5 px-3 text-sm text-gray-700">
                                            {result.label}
                                        </td>
                                        <td className="py-1.5 px-3 text-right text-sm text-gray-600">
                                            {(result.confidence * 100).toFixed(2)}%
                                        </td>
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </details>
                    )}

                    <div className="text-xs text-gray-500 mt-3">
                        Frames used: {classificationResults.frames_used}
                    </div>
                </div>
            )}
        </div>
    );
}
