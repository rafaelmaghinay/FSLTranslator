import React from 'react';

export default function NavigationBar({ onAboutClick, onHomeClick }) {
    return (
        <nav className="bg-white shadow-md fixed top-0 left-0 right-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16 items-center">
                    <div
                        className="flex items-center cursor-pointer gap-2"
                        onClick={onHomeClick}
                    >
                        <img
                            src="/logo.png"
                            alt="FSL Logo"
                            className="h-10 w-10"
                        />
                        <span className="text-2xl font-bold text-blue-600">FSL Recognizer</span>
                    </div>
                    <div className="flex space-x-6">
                        <button
                            onClick={onHomeClick}
                            className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
                        >
                            Home
                        </button>
                        <button
                            onClick={onAboutClick}
                            className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
                        >
                            About
                        </button>
                    </div>
                </div>
            </div>
        </nav>
    );
}
