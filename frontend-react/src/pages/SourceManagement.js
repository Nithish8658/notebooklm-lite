import React, { useState, useEffect, useCallback } from 'react';
import { fetchSources, deleteSource, clearAllSources, fetchMetrics, clearMetrics } from '../api';
import { useGlobalState } from '../GlobalState';

function SourceManagement() {
  const { user } = useGlobalState();
  const [sources, setSources] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [activeTab, setActiveTab] = useState('sources');
  const [loading, setLoading] = useState(false);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const sourceData = await fetchSources(user?.active_batch_id);
      setSources(sourceData);
      const metricsData = await fetchMetrics();
      setMetrics(metricsData);
    } catch (err) {
      console.error("Failed to load management data:", err);
    } finally {
      setLoading(false);
    }
  }, [user?.active_batch_id]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleDelete = async (docId) => {
    if (window.confirm("Permanently delete this source? This action cannot be undone.")) {
      try {
        await deleteSource(docId);
        setSources(prev => prev.filter(s => s.document_id !== docId));
      } catch (err) {
        alert("Failed to delete source: " + err.message);
      }
    }
  };

  const toggleStatus = async (docId, currentStatus) => {
    const endpoint = currentStatus ? 'deactivate' : 'activate';
    try {
      await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/sources/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_ids: [docId] })
      });
      loadData();
    } catch (err) {
      alert(`Failed to ${endpoint}: ` + err.message);
    }
  };

  const handleClearAll = async () => {
    if (window.confirm("CRITICAL: Wipe ENTIRE knowledge base?")) {
      try {
        await clearAllSources();
        setSources([]);
      } catch (err) {
        alert("Failed to clear: " + err.message);
      }
    }
  };

  const getTypeIcon = (type) => {
    const t = type.toLowerCase();
    if (t.includes('pdf')) return '📄';
    if (t.includes('youtube')) return '📺';
    if (t.includes('web')) return '🌐';
    if (t.includes('ppt')) return '📊';
    if (t.includes('xls')) return '📁';
    if (t.includes('mp4') || t.includes('video')) return '🎬';
    return '📝';
  };

  return (
    <div className="management-page" style={{ 
      padding: '40px', 
      maxWidth: '1200px', 
      margin: '0 auto', 
      flex: 1, 
      overflowY: 'auto' 
    }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '2rem' }}>Knowledge Base</h1>
          <p style={{ color: '#666', margin: '5px 0' }}>Manage your ingested sources and system performance.</p>
        </div>
        <div style={{ display: 'flex', gap: '15px' }}>
          <button onClick={loadData} className="btn-secondary" disabled={loading}>
            {loading ? "Refreshing..." : "Refresh"}
          </button>
          <button onClick={handleClearAll} className="btn-danger">Wipe System</button>
        </div>
      </header>

      <div className="tab-container" style={{ display: 'flex', gap: '20px', borderBottom: '1px solid #ddd', marginBottom: '30px' }}>
        <button 
          onClick={() => setActiveTab('sources')}
          style={{
            padding: '10px 20px',
            background: 'none',
            border: 'none',
            borderBottom: activeTab === 'sources' ? '3px solid #007bff' : '3px solid transparent',
            color: activeTab === 'sources' ? '#007bff' : '#666',
            cursor: 'pointer', fontWeight: 'bold'
          }}
        >
          Sources ({sources.length})
        </button>
        <button 
          onClick={() => setActiveTab('usage')}
          style={{
            padding: '10px 20px',
            background: 'none',
            border: 'none',
            borderBottom: activeTab === 'usage' ? '3px solid #007bff' : '3px solid transparent',
            color: activeTab === 'usage' ? '#007bff' : '#666',
            cursor: 'pointer', fontWeight: 'bold'
          }}
        >
          Performance & Cost
        </button>
      </div>

      {activeTab === 'sources' ? (
        <div className="source-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '20px' }}>
          {sources.length === 0 ? (
            <div style={{ gridColumn: '1/-1', textAlign: 'center', padding: '100px', background: '#f9f9f9', borderRadius: '12px' }}>
              No sources found. Add content from the Chat page.
            </div>
          ) : (
            sources.map(s => (
              <div key={s.document_id} className="source-card" style={{ 
                background: 'white', 
                borderRadius: '12px', 
                padding: '20px', 
                boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                border: s.is_active ? '1px solid #e1f5fe' : '1px solid #f5f5f5',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between'
              }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <span style={{ fontSize: '1.5rem' }}>{getTypeIcon(s.type)}</span>
                    <span style={{ 
                      fontSize: '0.7rem', 
                      padding: '3px 8px', 
                      borderRadius: '10px',
                      background: s.is_active ? '#4caf50' : '#9e9e9e',
                      color: 'white',
                      textTransform: 'uppercase',
                      fontWeight: 'bold'
                    }}>
                      {s.is_active ? 'Active' : 'Offline'}
                    </span>
                  </div>
                  <h4 style={{ margin: '15px 0 5px 0', wordBreak: 'break-all', fontSize: '1rem' }}>{s.filename}</h4>
                  <p style={{ fontSize: '0.75rem', color: '#888', margin: 0 }}>ID: {s.document_id.split('-')[0]}... | Type: {s.type}</p>
                  <p style={{ fontSize: '0.75rem', color: '#aaa', marginTop: '5px' }}>
                    Added: {new Date(s.ingested_at * 1000).toLocaleDateString()}
                  </p>
                </div>
                
                <div style={{ display: 'flex', gap: '10px', marginTop: '20px', paddingTop: '15px', borderTop: '1px solid #f0f0f0' }}>
                  <button 
                    onClick={() => toggleStatus(s.document_id, s.is_active)}
                    style={{ 
                      flex: 1, 
                      padding: '8px', 
                      borderRadius: '6px', 
                      border: '1px solid #ddd',
                      background: s.is_active ? '#fff' : '#f0f7ff',
                      color: s.is_active ? '#666' : '#007bff',
                      cursor: 'pointer',
                      fontSize: '0.85rem'
                    }}
                  >
                    {s.is_active ? 'Deactivate' : 'Bring Online'}
                  </button>
                  <button 
                    onClick={() => handleDelete(s.document_id)}
                    style={{ 
                      padding: '8px 15px', 
                      borderRadius: '6px', 
                      border: '1px solid #ffcdd2',
                      background: '#fff',
                      color: '#f44336',
                      cursor: 'pointer',
                      fontSize: '0.85rem'
                    }}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      ) : (
        <div className="metrics-dashboard">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '30px' }}>
            <div className="metric-card" style={{ background: '#f0f7ff', padding: '25px', borderRadius: '12px', textAlign: 'center' }}>
              <h3 style={{ margin: 0, fontSize: '2.5rem', color: '#007bff' }}>{metrics?.summary?.total_calls || 0}</h3>
              <p style={{ margin: '5px 0 0 0', color: '#666' }}>Gemini Calls</p>
            </div>
            <div className="metric-card" style={{ background: '#f6ffed', padding: '25px', borderRadius: '12px', textAlign: 'center' }}>
              <h3 style={{ margin: 0, fontSize: '2.5rem', color: '#52c41a' }}>{(metrics?.summary?.total_tokens || 0).toLocaleString()}</h3>
              <p style={{ margin: '5px 0 0 0', color: '#666' }}>Tokens Used</p>
            </div>
            <div className="metric-card" style={{ background: '#fff7e6', padding: '25px', borderRadius: '12px', textAlign: 'center' }}>
              <h3 style={{ margin: 0, fontSize: '2.5rem', color: '#fa8c16' }}>${metrics?.summary?.estimated_cost_usd || '0.00'}</h3>
              <p style={{ margin: '5px 0 0 0', color: '#666' }}>Estimated Cost</p>
            </div>
          </div>
          <div style={{ background: '#f9f9f9', padding: '20px', borderRadius: '12px', color: '#666', fontSize: '0.9rem' }}>
            <strong>Optimization Status:</strong> Gemini 2.0 Flash is active. Cross-Encoder reranking is enabled for precision. Caching is active on the Vector DB layer.
            <button 
              onClick={async () => { if(window.confirm("Reset?")) await clearMetrics(); loadData(); }}
              style={{ display: 'block', marginTop: '15px', color: '#999', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}
            >
              Reset usage statistics
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default SourceManagement;
