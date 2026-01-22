import React from 'react';
import BackButton from './BackButton.jsx';

export default function About({ onBack }) {
    const objectives = [
        "Design and implement a system using a YOLOv12-assisted Convolutional Neural Network (CNN) for Filipino Sign Language gesture recognition.",
        "Leverage transfer learning to improve the model's performance and accuracy.",
        "Compare the results of the proposed model with those from the study 'Filipino Sign Language Recognition Using Long Short-Term Memory and Residual Network Architecture', in order to identify potential improvements in recognition accuracy, efficiency, or overall system performance.",
        "Translate the recognized FSL gestures into real-time text outputs.",
        "Deploy the system as an accessible and user-friendly web application to facilitate real-time FSL communication."
    ];

    const recognizedGestures = {
        alphabet: "A-Z",
        digits: "1-9",
        phrases: [
            'Good Afternoon',
            'Good Evening',
            'Good Morning',
            'How Are You',
            "I'm Fine",
            'Sorry',
            'Thank You',
            "You're Welcome"
        ]
    };

    return (
        <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-8">
            <div className="text-center mb-8">
                <img
                    src="/logo.png"
                    alt="Filipino Sign Language Logo"
                    className="w-48 h-48 mx-auto mb-4"
                />
                <h2 className="text-3xl font-bold text-gray-900 mb-2">
                    Filipino Sign Language Recognition System
                </h2>
                <p className="text-lg text-gray-600">
                    YOLOv12-Assisted CNN for Real-Time FSL Translation
                </p>
            </div>

            <div className="mb-8">
                <h3 className="text-2xl font-semibold text-gray-900 mb-4 flex items-center">
                    <span className="mr-2">🎯</span>
                    Objectives
                </h3>
                <ol className="list-decimal list-inside space-y-3">
                    {objectives.map((obj, idx) => (
                        <li key={idx} className="text-gray-700 leading-relaxed pl-2">
                            {obj}
                        </li>
                    ))}
                </ol>
            </div>

            <div className="mb-8">
                <h3 className="text-2xl font-semibold text-gray-900 mb-4 flex items-center">
                    <span className="mr-2">🤝</span>
                    Recognized Gestures
                </h3>

                <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-blue-50 rounded-lg p-4">
                        <h4 className="font-semibold text-lg text-blue-900 mb-2">
                            Alphabet & Numbers
                        </h4>
                        <p className="text-gray-700">
                            <span className="font-medium">Alphabet:</span> {recognizedGestures.alphabet}
                        </p>
                        <p className="text-gray-700">
                            <span className="font-medium">Digits:</span> {recognizedGestures.digits}
                        </p>
                    </div>

                    <div className="bg-green-50 rounded-lg p-4">
                        <h4 className="font-semibold text-lg text-green-900 mb-2">
                            Common Phrases
                        </h4>
                        <ul className="text-gray-700 space-y-1">
                            {recognizedGestures.phrases.map((phrase, idx) => (
                                <li key={idx} className="flex items-center">
                                    <span className="mr-2">•</span>
                                    {phrase}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            </div>

            <div className="mb-8">
                <h3 className="text-2xl font-semibold text-gray-900 mb-4 flex items-center">
                    <span className="mr-2">🔬</span>
                    Technology Stack
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-gray-50 rounded-lg p-3 text-center">
                        <p className="font-semibold text-gray-900">YOLOv12</p>
                        <p className="text-sm text-gray-600">Detection</p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3 text-center">
                        <p className="font-semibold text-gray-900">ResNet-34 + BiLstm</p>
                        <p className="text-sm text-gray-600">Classification</p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3 text-center">
                        <p className="font-semibold text-gray-900">React</p>
                        <p className="text-sm text-gray-600">JavaScript</p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3 text-center">
                        <p className="font-semibold text-gray-900">FastApi</p>
                        <p className="text-sm text-gray-600">Python</p>
                    </div>
                </div>
            </div>

            <div className="flex justify-center">
                <BackButton onClick={onBack} />
            </div>
        </div>
    );
}
