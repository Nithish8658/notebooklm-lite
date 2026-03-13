import { useState } from "react";
import { uploadDocument } from "../api";
import { useGlobalState } from "../GlobalState";

function VideoUpload() {
  const { videoStatus: status, setVideoStatus: setStatus } = useGlobalState();
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const allowedExtensions = [".mp4", ".mkv", ".mov", ".avi", ".webm"];
      const isAllowed = allowedExtensions.some(ext => selectedFile.name.toLowerCase().endsWith(ext));

      if (!isAllowed) {
        setStatus("Error: Only video files (MP4, MKV, MOV, AVI, WEBM) are allowed.");
        setFile(null);
        e.target.value = null; 
        return;
      }
      setFile(selectedFile);
      setStatus("");
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setStatus("No video file selected.");
      return;
    }

    try {
      setLoading(true);
      setStatus("Uploading and processing video (this may take several minutes)...");
      const result = await uploadDocument(file);
      setStatus(`Success: ${result.filename} indexed!`);
      setFile(null);
    } catch (err) {
      setStatus(`Upload failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container video-upload-container">
      <h3>Upload Video Course</h3>

      <div className="upload-controls">
        <input
          type="file"
          accept=".mp4,.mkv,.mov,.avi,.webm"
          onChange={handleFileChange}
          disabled={loading}
        />
        <button onClick={handleUpload} disabled={loading || !file}>
          {loading ? "Processing..." : "Upload Video"}
        </button>
      </div>

      {status && (
        <div className={`upload-status ${status.startsWith("Success") ? "success" : "error"}`}>
          {status}
        </div>
      )}
    </div>
  );
}

export default VideoUpload;
