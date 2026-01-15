import React, { useState, useEffect } from 'react';

export default function ImageSlider({ images = [], className = '', maxHeight = 'max-h-72', onIndexChange }) {
    const imgList = Array.isArray(images) ? images : images ? [images] : [];
    const [current, setCurrent] = useState(0);

    // avoid unconditional setState inside effect (ESLint rule)
    useEffect(() => {
        if (current !== 0) setCurrent(0);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [images]);

    useEffect(() => {
        if (typeof onIndexChange === 'function') onIndexChange(current);
    }, [current, onIndexChange]);

    const showPrev = () => {
        if (!imgList.length) return;
        setCurrent((c) => (c - 1 + imgList.length) % imgList.length);
    };

    const showNext = () => {
        if (!imgList.length) return;
        setCurrent((c) => (c + 1) % imgList.length);
    };

    if (!imgList.length) {
        return <div className={`text-gray-600 ${className}`}>No images to preview</div>;
    }

    return (
        <div className={className}>
            <div className="relative">
                <img
                    src={imgList[current]}
                    alt={`Preview ${current + 1}`}
                    className={`mx-auto w-full object-contain rounded bg-gray-50 ${maxHeight}`}
                />

                {imgList.length > 1 && (
                    <>
                        <button
                            onClick={showPrev}
                            aria-label="Previous"
                            className="absolute left-2 top-1/2 -translate-y-1/2 bg-white/90 hover:bg-white px-2 py-1 rounded-full shadow"
                        >
                            ‹
                        </button>

                        <button
                            onClick={showNext}
                            aria-label="Next"
                            className="absolute right-2 top-1/2 -translate-y-1/2 bg-white/90 hover:bg-white px-2 py-1 rounded-full shadow"
                        >
                            ›
                        </button>

                        <div className="flex justify-center gap-2 mt-3">
                            {imgList.map((_, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => setCurrent(idx)}
                                    aria-label={`Go to ${idx + 1}`}
                                    className={`w-3 h-3 rounded-full transition-colors ${idx === current ? 'bg-green-600' : 'bg-gray-300'}`}
                                />
                            ))}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}