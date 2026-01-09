import React from 'react';

const NavigationBar = () => {
    return (
        <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between p-4 bg-white shadow">
            <a href="/" aria-label="Home" className="flex items-center">
                <img
                    src="/logo.png"
                    alt="Logo"
                    className="w-8 h-8 sm:w-10 sm:h-10 object-contain inline-block"
                />
                <span className="hidden sm:inline-block ml-2">
                    Filipino Sign Language Recognizer AI
                </span>
            </a>

            <div className="flex space-x-4">
                <a href="/" className="text-gray-700 hover:text-gray-900">Home</a>
                <a href="/events" className="text-gray-700 hover:text-gray-900">About us</a>
            </div>
        </nav>
    );
};

export default NavigationBar;
