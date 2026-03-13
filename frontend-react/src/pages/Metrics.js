import React, { useState, useEffect } from 'react';
import { fetchMetrics, clearMetrics } from '../api';

function Metrics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadMetrics = async () => {
    try {
      setLoading(true);
      const data = await fetchMetrics();
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMetrics();
    const interval = setInterval(loadMetrics, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const handleClear = async () => {
    if (window.confirm("Are you sure you want to clear all historical metrics?")) {
      try {
        await clearMetrics();
        loadMetrics();
      } catch (err) {
        alert("Failed to clear metrics: " + err.message);
      }
    }
  };

  if (loading && !metrics) return <div className="studio-tool"><p>Loading metrics...</p></div>;

  return (
    <div className="studio-tool" style={{ padding: '30px' }}>
      <div className="tool-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>System Performance & Logs</h2>
          <p>Monitor LLM usage, latency, and hardware stability.</p>
        </div>
        <button onClick={handleClear} className="control-btn secondary" style={{ background: '#fff1f0', color: '#f5222d', border: '1px solid #ffa39e' }}>
          Clear Logs
        </button>
      </div>

      {error && <div className="upload-status error">{error}</div>}

      {metrics && (
        <div style={{ marginTop: '30px' }}>
          {/* LLM Dashboard */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '40px' }}>
            <div className="metric-card" style={cardStyle}>
              <div style={labelStyle}>Total LLM Calls</div>
              <div style={valueStyle}>{metrics.summary.total_calls}</div>
            </div>
            <div className="metric-card" style={cardStyle}>
              <div style={labelStyle}>Total Tokens</div>
              <div style={valueStyle}>{metrics.summary.total_tokens.toLocaleString()}</div>
            </div>
            <div className="metric-card" style={cardStyle}>
              <div style={labelStyle}>Prompt / Completion</div>
              <div style={valueStyle}>{metrics.summary.prompt_tokens.toLocaleString()} / {metrics.summary.candidates_tokens.toLocaleString()}</div>
            </div>
            <div className="metric-card" style={{ ...cardStyle, background: '#f6ffed', border: '1px solid #b7eb8f' }}>
              <div style={labelStyle}>Est. Cost (USD)</div>
              <div style={{ ...valueStyle, color: '#52c41a' }}>${metrics.summary.estimated_cost_usd.toFixed(4)}</div>
            </div>
          </div>

          {/* Detailed Logs */}
          <div style={{ background: 'white', borderRadius: '12px', border: '1px solid #eaeaea', overflow: 'hidden' }}>
            <div style={{ padding: '15px 20px', background: '#fafafa', borderBottom: '1px solid #eaeaea', fontWeight: '600' }}>
              Raw Event Logs (Last 100)
            </div>
            <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                <thead style={{ position: 'sticky', top: 0, background: '#f0f2f5', zIndex: 1 }}>
                  <tr>
                    <th style={thStyle}>Category</th>
                    <th style={thStyle}>Operation</th>
                    <th style={thStyle}>Duration</th>
                    <th style={thStyle}>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.raw.slice().reverse().map((event, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid #f0f0f0' }}>
                      <td style={tdStyle}>
                        <span style={{ ...tagStyle, background: getCategoryColor(event.category) }}>
                          {event.category}
                        </span>
                      </td>
                      <td style={tdStyle}>{event.operation}</td>
                      <td style={tdStyle}>{event.duration_ms ? `${event.duration_ms}ms` : '-'}</td>
                      <td style={{ ...tdStyle, fontSize: '0.8rem', color: '#666', fontFamily: 'monospace' }}>
                        {JSON.stringify(event.metadata)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const cardStyle = {
  padding: '20px',
  background: '#fff',
  borderRadius: '12px',
  border: '1px solid #eaeaea',
  boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
};

const labelStyle = {
  fontSize: '0.85rem',
  color: '#8c8c8c',
  marginBottom: '8px',
  textTransform: 'uppercase',
  letterSpacing: '0.5px'
};

const valueStyle = {
  fontSize: '1.5rem',
  fontWeight: '700',
  color: '#262626'
};

const thStyle = {
  textAlign: 'left',
  padding: '12px 20px',
  borderBottom: '1px solid #eaeaea',
  color: '#595959',
  fontWeight: '600'
};

const tdStyle = {
  padding: '12px 20px',
  verticalAlign: 'top'
};

const tagStyle = {
  padding: '2px 8px',
  borderRadius: '4px',
  color: 'white',
  fontSize: '0.75rem',
  fontWeight: '600',
  textTransform: 'uppercase'
};

function getCategoryColor(cat) {
  switch (cat) {
    case 'llm': return '#1890ff';
    case 'retrieval': return '#722ed1';
    case 'ingestion': return '#faad14';
    case 'system': return '#52c41a';
    case 'performance': return '#eb2f96';
    case 'error': return '#f5222d';
    case 'content': return '#13c2c2';
    default: return '#bfbfbf';
  }
}

export default Metrics;
