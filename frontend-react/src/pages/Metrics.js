import React, { useState, useEffect } from 'react';
import { fetchMetrics, clearMetrics } from '../api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

function Metrics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('24h');

  const loadMetrics = async () => {
    try {
      setLoading(true);
      const data = await fetchMetrics(timeRange);
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
    const interval = setInterval(loadMetrics, 15000); // Refresh every 15s
    return () => clearInterval(interval);
  }, [timeRange]);

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
        <div style={{ display: 'flex', gap: '10px' }}>
          <select 
            value={timeRange} 
            onChange={(e) => setTimeRange(e.target.value)}
            style={{ padding: '8px 12px', borderRadius: '6px', border: '1px solid #d9d9d9', outline: 'none' }}
          >
            <option value="1h">Last 1 Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          <button onClick={handleClear} className="control-btn secondary" style={{ background: '#fff1f0', color: '#f5222d', border: '1px solid #ffa39e' }}>
            Clear Logs
          </button>
        </div>
      </div>

      {error && <div className="upload-status error">{error}</div>}

      {metrics && (
        <div style={{ marginTop: '30px' }}>
          {/* Charts Row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '30px' }}>
            <div style={{ background: 'white', padding: '20px', borderRadius: '12px', border: '1px solid #eaeaea' }}>
              <h4 style={{ margin: '0 0 20px 0', fontSize: '0.9rem', color: '#595959' }}>Requests Per {timeRange === '1h' ? 'Minute' : (timeRange === '24h' ? 'Hour' : 'Day')}</h4>
              <div style={{ height: '250px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={metrics.chart_data}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                    <XAxis dataKey="time" fontSize={10} tickMargin={10} />
                    <YAxis fontSize={10} />
                    <Tooltip />
                    <Area type="monotone" dataKey="calls" stroke="#1890ff" fill="#e6f7ff" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div style={{ background: 'white', padding: '20px', borderRadius: '12px', border: '1px solid #eaeaea' }}>
              <h4 style={{ margin: '0 0 20px 0', fontSize: '0.9rem', color: '#595959' }}>Tokens Per {timeRange === '1h' ? 'Minute' : (timeRange === '24h' ? 'Hour' : 'Day')}</h4>
              <div style={{ height: '250px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={metrics.chart_data}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                    <XAxis dataKey="time" fontSize={10} tickMargin={10} />
                    <YAxis fontSize={10} />
                    <Tooltip />
                    <Area type="monotone" dataKey="tokens" stroke="#52c41a" fill="#f6ffed" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* LLM Dashboard */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '40px' }}>
            <div className="metric-card" style={cardStyle}>
              <div style={labelStyle}>Total LLM Calls</div>
              <div style={valueStyle}>{metrics.summary?.total_calls || 0}</div>
            </div>
            <div className="metric-card" style={cardStyle}>
              <div style={labelStyle}>Total Tokens</div>
              <div style={valueStyle}>{(metrics.summary?.total_tokens || 0).toLocaleString()}</div>
            </div>
            <div className="metric-card" style={cardStyle}>
              <div style={labelStyle}>Prompt / Completion</div>
              <div style={valueStyle}>{(metrics.summary?.prompt_tokens || 0).toLocaleString()} / {(metrics.summary?.candidates_tokens || 0).toLocaleString()}</div>
            </div>
            <div className="metric-card" style={{ ...cardStyle, background: '#f6ffed', border: '1px solid #b7eb8f' }}>
              <div style={labelStyle}>Est. Cost (USD)</div>
              <div style={{ ...valueStyle, color: '#52c41a' }}>${(metrics.summary?.estimated_cost_usd || 0).toFixed(4)}</div>
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
