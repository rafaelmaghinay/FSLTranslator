import React, { useState } from 'react';
import NavigationBar from './components/NavigationBar.jsx';
import Upload from './components/Upload.jsx';
import StartAnalysis from './components/StartAnalysis.jsx';
import Results from './components/Results.jsx';
import LiveCamera from './components/LiveCamera.jsx';
import About from './components/About.jsx';

function App() {
    const [view, setView] = useState('upload');
    const [uploadedData, setUploadedData] = useState(null);
    const [result, setResult] = useState(null);

    const reset = () => {
        setView('upload');
        setUploadedData(null);
        setResult(null);
    };

    const handleUploaded = (data) => {
        setUploadedData(data);
        setView('start');
    };

    const handleResults = (res) => {
        setResult(res);
        setView('results');
    };

    const handleStartLive = () => {
        setView('live');
    };

    const handleAbout = () => {
        setView('about');
    };

    return (
        <div className="min-h-screen bg-gray-50">
            <NavigationBar onAboutClick={handleAbout} onHomeClick={reset} />

            <div className="pt-20 px-4 pb-8">
                <h1 className="text-4xl font-bold text-center text-gray-900 mb-2">
                    FSL Recognition System
                </h1>
                <p className="text-center text-gray-600 mb-8">
                    Filipino Sign Language Translation
                </p>

                {view === 'upload' && (
                    <Upload
                        onUploaded={handleUploaded}
                        onStartLive={handleStartLive}
                    />
                )}

                {view === 'start' && uploadedData && (
                    <StartAnalysis
                        urls={uploadedData.urls}
                        paths={uploadedData.paths}
                        onResults={handleResults}
                        onBack={reset}
                    />
                )}

                {view === 'results' && (
                    <Results
                        urls={uploadedData?.urls || []}
                        result={result}
                        onBack={reset}
                    />
                )}

                {view === 'live' && (
                    <LiveCamera onBack={reset} />
                )}

                {view === 'about' && (
                    <About onBack={reset} />
                )}
            </div>
        </div>
    );
}

export default App;
