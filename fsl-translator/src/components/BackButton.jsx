import React from 'react';

export default function BackButton({ onClick, label = 'Home', disabled = false }) {
    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className="px-4 py-2 bg-gray-300 hover:bg-gray-400 text-gray-800 font-semibold rounded disabled:opacity-60"
        >
            {label}
        </button>
    );
}