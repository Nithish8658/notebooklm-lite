import { useState, useEffect } from "react";
import { ingestBatchUrls } from "../api";
import { useGlobalState } from "../GlobalState";

function S3PresignedIngest() {
  const { user, setWebStatus: setStatus } = useGlobalState();
  const [urlsText, setUrlsText] = useState("");
  const [cohortId, setCohortId] = useState(user?.active_cohort_id || "default_cohort");
  const [loading, setLoading] = useState(false);
  const [localStatus, setLocalStatus] = useState("");

  // Sync cohortId if active context changes
  useEffect(() => {
    if (user?.active_cohort_id) {
      setCohortId(user.active_cohort_id);
    }
  }, [user?.active_cohort_id]);

  const handleIngest = async () => {
    const urls = urlsText
      .split("\n")
      .map((u) => u.trim())
      .filter((u) => u.startsWith("http"));

    if (urls.length === 0) {
      setLocalStatus("Please enter at least one valid URL.");
      return;
    }

    try {
      setLoading(true);
      setLocalStatus(`Starting batch ingestion for ${urls.length} files...`);
      
      await ingestBatchUrls(urls, cohortId);
      
      setLocalStatus(`Success: ${urls.length} S3 sources indexed!`);
      setStatus(`Success: ${urls.length} S3 sources indexed!`);
      setUrlsText("");
    } catch (err) {
      setLocalStatus(`Batch ingestion failed: ${err.message}`);
      setStatus(`Batch ingestion failed`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <h3>Add S3 Presigned URLs (Batch)</h3>

      <div style={{ marginBottom: '10px' }}>
        <label style={{ fontSize: '0.8rem', color: '#666', display: 'block', marginBottom: '4px' }}>Target Cohort ID</label>
        <input 
          type="text" 
          value={cohortId} 
          onChange={(e) => setCohortId(e.target.value)}
          placeholder="Cohort ID"
          style={{ 
            width: '100%', 
            padding: '6px', 
            borderRadius: '4px', 
            border: '1px solid #ddd',
            fontSize: '0.8rem'
          }}
        />
      </div>

      <p style={{ fontSize: '0.8rem', color: '#666', marginBottom: '10px' }}>
        Paste one or more presigned URLs below (one per line).
      </p>

      <div className="upload-controls" style={{ flexDirection: 'column', gap: '10px' }}>
        <textarea
          placeholder="https://bucket.s3.amazonaws.com/file.pdf?AWSAccessKeyId=..."
          value={urlsText}
          onChange={(e) => setUrlsText(e.target.value)}
          disabled={loading}
          rows={5}
          style={{ 
            width: '100%', 
            padding: '10px', 
            borderRadius: '8px', 
            border: '1px solid #ddd',
            fontSize: '0.85rem',
            fontFamily: 'monospace',
            resize: 'vertical'
          }}
        />
        <button 
          onClick={handleIngest} 
          disabled={loading || !urlsText.trim()}
          style={{ width: '100%' }}
        >
          {loading ? "Processing Batch..." : `Ingest ${urlsText.split("\n").filter(u => u.trim().startsWith("http")).length} URLs`}
        </button>
      </div>

      {localStatus && (
        <div className={`upload-status ${localStatus.startsWith("Success") ? "success" : "error"}`}>
          {localStatus}
        </div>
      )}
    </div>
  );
}

export default S3PresignedIngest;
