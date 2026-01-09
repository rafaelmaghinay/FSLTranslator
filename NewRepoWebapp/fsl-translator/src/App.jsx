import './App.css'
import NavigationBar from "./components/NavigationBar.jsx";
import { useState } from "react";

function App() {

    const API = "http://127.0.0.1:8000";
    const [status, setStatus] = useState("");     // message text
    const [isUploading, setIsUploading] = useState(false);


    const cards = [
        { title: "Upload Video", icon: "🎥" },
        { title: "Upload Image", icon: "🖼️"},
        { title: "Upload Sequence of Images", icon: "📁"  },
        { title: "Start Live Camera", icon: "📹" }
    ];

    const handleUploadVideo = () => {
        document.getElementById('videoInput').click();
    };

    const handleUploadImage = () => {
        document.getElementById('imageInput').click();
    };

    const handleUploadMultiple = () => {
        document.getElementById('multipleInput').click();
    };

    const handleLiveCamera = () => {
        console.log('Starting live camera...');
        // Add camera functionality here
    };

    const buttonHandlers = [handleUploadVideo, handleUploadImage, handleUploadMultiple, handleLiveCamera];

    async function uploadSingleImage(file) {
    const fd = new FormData();
    fd.append("image", file);

    const res = await fetch(`${API}/api/upload/image`, {
        method: "POST",
        body: fd,
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data?.detail || "Image upload failed");
    return data;
    }

    async function uploadSequence(files) {
    const fd = new FormData();
    [...files].forEach((f) => fd.append("images", f));

    const res = await fetch(`${API}/api/upload/sequence`, {
        method: "POST",
        body: fd,
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data?.detail || "Sequence upload failed");
    return data;
    }

    async function uploadVideo(file) {
    const fd = new FormData();
    fd.append("video", file);

    const res = await fetch(`${API}/api/upload/video`, {
        method: "POST",
        body: fd,
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data?.detail || "Video upload failed");
    return data;
    }



    return (
      <main>
          <div className="wrapper">
              <header className="App-header">
                  <NavigationBar />
              </header>

              <section className="hero py-12 bg-gray-50 mt-10">
                  <div className="max-w-3xl mx-auto text-center px-4">
                      <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900">
                          Filipino Sign Language Recognizer AI
                      </h2>
                      <p className="mt-4 text-lg text-gray-600">
                          AI-powered Sign Language Recognizer for Filipino Sign Language
                      </p>

                      <div className="mt-12 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4. gap-4 max-w-2xl mx-auto">
                          {cards.map((card, index) => (
                              <div key={index} className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow">
                                  <div className="text-3xl mb-3">{card.icon}</div>
                                  <h3 className="text-xl font-semibold text-gray-900">{card.title}</h3>
                                  <button
                                    onClick={buttonHandlers[index]}
                                    disabled={isUploading}
                                    className={`mt-4 w-full font-semibold py-1 px-2 rounded transition-colors text-sm
                                        ${isUploading ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700 text-white"}`}
                                    >
                                    {isUploading ? "Uploading..." : card.title}
                                  </button>
                              </div>
                          ))}
                      </div>
                  </div>
              </section>
              <input
  type="file"
  id="videoInput"
  accept="video/*"
  hidden
  onChange={async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setIsUploading(true);
      setStatus(`Uploading video: ${file.name} ...`);
      const result = await uploadVideo(file);
      setStatus(`✅ Video uploaded: ${result.saved_as}`);
      console.log("VIDEO UPLOAD RESULT:", result);
    } catch (err) {
      setStatus(`❌ ${err.message}`);
      console.error(err);
    } finally {
      setIsUploading(false);
      e.target.value = ""; // reset so you can re-upload same file
    }
  }}
/>

<input
  type="file"
  id="imageInput"
  accept="image/*"
  hidden
  onChange={async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setIsUploading(true);
      setStatus(`Uploading image: ${file.name} ...`);
      const result = await uploadSingleImage(file);
      setStatus(`✅ Image uploaded: ${result.saved_as}`);
      console.log("IMAGE UPLOAD RESULT:", result);
    } catch (err) {
      setStatus(`❌ ${err.message}`);
      console.error(err);
    } finally {
      setIsUploading(false);
      e.target.value = "";
    }
  }}
/>

<input
  type="file"
  id="multipleInput"
  accept="image/*"
  multiple
  hidden
  onChange={async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    try {
      setIsUploading(true);
      setStatus(`Uploading ${files.length} images...`);
      const result = await uploadSequence(files);
      setStatus(`✅ Sequence uploaded: ${result.count} files (folder: ${result.folder})`);
      console.log("SEQUENCE UPLOAD RESULT:", result);
    } catch (err) {
      setStatus(`❌ ${err.message}`);
      console.error(err);
    } finally {
      setIsUploading(false);
      e.target.value = "";
    }
  }}
/>

          </div>
      </main>
  )


}

export default App
