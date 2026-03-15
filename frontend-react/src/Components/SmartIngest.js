import { useState, useEffect } from "react";
import { ingestSmartUrl } from "../api";
import { useGlobalState } from "../GlobalState";

function SmartIngest() {
  const { user, setWebStatus: setStatus } = useGlobalState();
  const [url, setUrl] = useState("");
  const [batchId, setBatchId] = useState(user?.active_batch_id || "default_batch");
  const [mode, setMode] = useState("single");
  const [loading, setLoading] = useState(false);
  const [localStatus, setLocalStatus] = useState("");

  // Sync batchId if active context changes
  useEffect(() => {
    if (user?.active_batch_id) {
      setBatchId(user.active_batch_id);
    }
  }, [user?.active_batch_id]);

  const handleIngest = async () => {
    if (!url) {
      setLocalStatus("Please enter a URL.");
      return;
    }

    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      setLocalStatus("Please enter a valid URL (starting with http/https).");
      return;
    }

    try {
      setLoading(true);
      const actionText = mode === "crawl" ? "Crawling" : mode === "sitemap" ? "Reading sitemap" : "Analyzing";
      setLocalStatus(`${actionText} and indexing content...`);
      
      await ingestSmartUrl(url, mode, batchId);
      
      setLocalStatus(`Success: ${url} indexed!`);
      setStatus(`Success: ${url} indexed!`);
      setUrl("");
    } catch (err) {
      setLocalStatus(`Ingestion failed: ${err.message}`);
      setStatus(`Smart ingestion failed for ${url}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <h3>Add URL (YouTube / Web / File)</h3>

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

      <div className="upload-controls" style={{ flexDirection: 'column', gap: '10px' }}>
        <div style={{ display: 'flex', gap: '10px', width: '100%' }}>
          <input
            type="text"
            placeholder="https://..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={loading}
            style={{ flex: 1 }}
          />
          <button onClick={handleIngest} disabled={loading || !url}>
            {loading ? "Adding..." : "Add"}
          </button>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.8rem', color: '#666' }}>
          <span>Web Mode:</span>
          <select 
            value={mode} 
            onChange={(e) => setMode(e.target.value)}
            disabled={loading}
            style={{ padding: '2px 5px', borderRadius: '4px', border: '1px solid #ddd' }}
          >
            <option value="single">Single Page</option>
            <option value="crawl">Depth-1 Crawl</option>
            <option value="sitemap">Sitemap.xml</option>
          </select>
          <span style={{ fontStyle: 'italic', fontSize: '0.75rem' }}>
            (Mode only applies to standard websites)
          </span>
        </div>
      </div>

      {localStatus && (
        <div className={`upload-status ${localStatus.startsWith("Success") ? "success" : "error"}`}>
          {localStatus}
        </div>
      )}
    </div>
  );
}

export default SmartIngest;
