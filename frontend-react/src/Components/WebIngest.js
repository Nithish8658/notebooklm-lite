import { useState, useEffect } from "react";
import { ingestWeb } from "../api";
import { useGlobalState } from "../GlobalState";

function WebIngest() {
  const { user, webStatus: status, setWebStatus: setStatus } = useGlobalState();
  const [url, setUrl] = useState("");
  const [batchId, setBatchId] = useState(user?.active_batch_id || "default_batch");
  const [mode, setMode] = useState("single");
  const [loading, setLoading] = useState(false);

  // Sync batchId if active context changes
  useEffect(() => {
    if (user?.active_batch_id) {
      setBatchId(user.active_batch_id);
    }
  }, [user?.active_batch_id]);

  const handleIngest = async () => {
    if (!url) {
      setStatus("Please enter a URL.");
      return;
    }

    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      setStatus("Please enter a valid URL (starting with http/https).");
      return;
    }

    try {
      setLoading(true);
      const actionText = mode === "crawl" ? "Crawling" : mode === "sitemap" ? "Reading sitemap" : "Scraping";
      setStatus(`${actionText} and indexing website...`);
      const result = await ingestWeb(url, mode, batchId);
      setStatus(`Success: ${mode === "single" ? "Website" : "Content"} indexed! (ID: ${result.document_id})`);
      setUrl("");
    } catch (err) {
      setStatus(`Ingestion failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container web-ingest-container">
      <h3>Add Website Content</h3>

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
          type="text"
          placeholder="https://example.com/docs"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={loading}
        />
        
        <select 
          value={mode} 
          onChange={(e) => setMode(e.target.value)}
          disabled={loading}
          className="mode-select"
        >
          <option value="single">Single Page</option>
          <option value="crawl">Depth-1 Crawl</option>
          <option value="sitemap">Sitemap.xml</option>
        </select>

        <button onClick={handleIngest} disabled={loading || !url}>
          {loading ? "Processing..." : "Add Content"}
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

export default WebIngest;