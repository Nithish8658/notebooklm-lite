import { useState, useEffect } from "react";
import { uploadDocument } from "../api";
import { useGlobalState } from "../GlobalState";

function Upload() {
  const { user, uploadStatus: status, setUploadStatus: setStatus } = useGlobalState();
  const [file, setFile] = useState(null);
  const [batchId, setBatchId] = useState(user?.active_batch_id || "default_batch");

  // Sync batchId if active context changes
  useEffect(() => {
    if (user?.active_batch_id) {
      setBatchId(user.active_batch_id);
    }
  }, [user?.active_batch_id]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const allowedTypes = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "video/mp4", "video/x-matroska", "video/quicktime", "video/x-msvideo", "video/webm"
      ];
      const allowedExtensions = [".pdf", ".pptx", ".ppt", ".xlsx", ".xls", ".xlsb", ".mp4", ".mkv", ".mov", ".avi", ".webm"];
      const isAllowed = allowedTypes.includes(selectedFile.type) || 
                        allowedExtensions.some(ext => selectedFile.name.toLowerCase().endsWith(ext));

      if (!isAllowed) {
        setStatus("Error: Only PDF, PPTX, Excel, and Video files are allowed.");
        setFile(null);
        e.target.value = null; // Reset input
        return;
      }
      setFile(selectedFile);
      setStatus("");
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setStatus("No file selected.");
      return;
    }

    try {
      setStatus("Uploading and analyzing file...");
      const result = await uploadDocument(file, batchId);
      setStatus(`Success: ${result.filename} indexed!`);
      setFile(null);
    } catch (err) {
      setStatus(`Upload failed: ${err.message}`);
    }
  };

  return (
    <div className="upload-container">
      <h3>Add Files (PDF / PPT / Video / Excel)</h3>

      <div style={{ marginBottom: '10px' }}>
        <label style={{ fontSize: '0.8rem', color: '#666', display: 'block', marginBottom: '4px' }}>Target Batch ID</label>
        <input 
          type="text" 
          value={batchId} 
          onChange={(e) => setBatchId(e.target.value)}
          placeholder="Batch ID"
          style={{ 
            width: '100%', 
            padding: '6px', 
            borderRadius: '4px', 
            border: '1px solid #ddd',
            fontSize: '0.8rem'
          }}
        />
      </div>

      <div className="upload-controls">
        <input
          type="file"
          accept=".pdf,.pptx,.ppt,.xlsx,.xls,.xlsb,.mp4,.mkv,.mov,.avi,.webm"
          onChange={handleFileChange}
        />
        <button onClick={handleUpload} disabled={!file}>
          Upload
        </button>
      </div>

      {status && (
        <div className={`upload-status ${status.startsWith("Uploaded") ? "success" : "error"}`}>
          {status}
        </div>
      )}
    </div>
  );
}

export default Upload;
