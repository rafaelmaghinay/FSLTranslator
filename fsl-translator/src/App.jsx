import './App.css'
import NavigationBar from "./components/NavigationBar.jsx";

function App() {
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
                                      className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-1 px-2 rounded transition-colors text-sm"
                                  >
                                      {card.title}
                                  </button>
                              </div>
                          ))}
                      </div>
                  </div>
              </section>
              <input type="file" id="videoInput" accept="video/*" hidden onChange={(e) => console.log('Video selected:', e.target.files)} />
              <input type="file" id="imageInput" accept="image/*" hidden onChange={(e) => console.log('Image selected:', e.target.files)} />
              <input type="file" id="multipleInput" accept="image/*" multiple hidden onChange={(e) => console.log('Images selected:', e.target.files)} />
          </div>
      </main>
  )
}

export default App
