// File: `src/App.jsx`
import './App.css';
import NavigationBar from './components/NavigationBar.jsx';
import Upload from './components/Upload.jsx';
import StartAnalysis from './components/StartAnalysis.jsx';
import Results from './components/Results.jsx';
import LiveCamera from './components/LiveCamera.jsx';
import { useState } from 'react';

export default function App() {
    const [uploaded, setUploaded] = useState(null); // { urls: [], paths: [], type }
    const [result, setResult] = useState(null);
    const [liveActive, setLiveActive] = useState(false);

    const handleUploaded = (payload) => {
        setUploaded(payload);
        setResult(null);
    };

    const handleStartResults = (res) => {
        setResult(res);
    };

    const handleClearAll = () => {
        setUploaded(null);
        setResult(null);
        setLiveActive(false);
    };

    const handleStartLive = () => {
        setUploaded(null);
        setResult(null);
        setLiveActive(true);
    };

    const handleLiveBack = () => {
        // called when LiveCamera signals back; clear live mode and any results
        setLiveActive(false);
        setUploaded(null);
        setResult(null);
    };

    return (
        <main>
            <div className="wrapper">
                <header className="App-header">
                    <NavigationBar />
                </header>

                <section className="hero py-12 bg-gray-50 mt-10">
                    <div className="max-w-5xl mx-auto text-center px-4">
                        <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900">
                            Filipino Sign Language Recognizer AI
                        </h2>
                        <p className="mt-4 text-lg text-gray-600">AI-powered Sign Language Recognizer for Filipino Sign Language</p>

                        <div className="mt-12">
                            {liveActive ? (
                                <LiveCamera onResults={handleStartResults} onBack={handleLiveBack} />
                            ) : (
                                !uploaded && !result ? (
                                    <Upload onUploaded={handleUploaded} onStartLive={handleStartLive} />
                                ) : uploaded && !result ? (
                                    <StartAnalysis urls={uploaded.urls} paths={uploaded.paths} onResults={handleStartResults} onBack={handleClearAll} />
                                ) : result ? (
                                    <Results urls={uploaded ? uploaded.urls : []} result={result} onBack={handleClearAll} />
                                ) : null
                            )}
                        </div>
                    </div>
                </section>
            </div>
        </main>
    );
}