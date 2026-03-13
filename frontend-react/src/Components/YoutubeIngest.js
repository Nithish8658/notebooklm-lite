import { useState, useEffect } from "react";
import { ingestYoutube } from "../api";
import { useGlobalState } from "../GlobalState";

function YoutubeIngest() {
  const { user, youtubeStatus: status, setYoutubeStatus: setStatus } = useGlobalState();
  const [url, setUrl] = useState("");
  const [cohortId, setCohortId] = useState(user?.active_cohort_id || "default_cohort");
  const [loading, setLoading] = useState(false);

  // Sync cohortId if active context changes
  useEffect(() => {
    if (user?.active_cohort_id) {
      setCohortId(user.active_cohort_id);
    }
  }, [user?.active_cohort_id]);

  const handleIngest = async () => {
    if (!url) {
      setStatus("Please enter a YouTube URL.");
      return;
    }

    // Basic URL validation
    if (!url.includes("youtube.com") && !url.includes("youtu.be")) {
      setStatus("Please enter a valid YouTube URL.");
      return;
    }

    try {
      setLoading(true);
      setStatus("Processing YouTube video (this may take a minute)...");
      const result = await ingestYoutube(url, cohortId);
      setStatus(`Success: Video indexed! (ID: ${result.document_id})`);
      setUrl("");
    } catch (err) {
      setStatus(`Ingestion failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container youtube-ingest-container">
      <h3>Add YouTube Video</h3>

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

      <div className="upload-controls">
        <input
          type="text"
          placeholder="https://www.youtube.com/watch?v=..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={loading}
        />
        <button onClick={handleIngest} disabled={loading || !url}>
          {loading ? "Processing..." : "Add Video"}
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

export default YoutubeIngest;
