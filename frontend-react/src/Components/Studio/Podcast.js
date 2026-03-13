import React, { useState } from 'react';
import { useGlobalState } from '../../GlobalState';

const BASE_URL = "http://127.0.0.1:8000";

function Podcast() {
  const { podcasts } = useGlobalState();
  const [activePodcast, setActivePodcast] = useState(null);

  const handleDownload = (pod) => {
    const link = document.createElement('a');
    link.href = `${BASE_URL}/podcasts/${pod.audio_path}`;
    link.download = `${pod.topic.replace(/\s+/g, '_')}.mp3`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const podcastList = Array.isArray(podcasts) ? podcasts : [];

  return (
    <div className="studio-tool">
      <div className="tool-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>Audio Overview</h2>
          <p>Listen to deep-dive conversations about your sources.</p>
        </div>
        {activePodcast && activePodcast.audio_path && (
          <button 
            onClick={() => handleDownload(activePodcast)}
            className="control-btn secondary"
            style={{ fontSize: '0.8rem', padding: '8px 15px' }}
          >
            Download MP3
          </button>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '30px', marginTop: '30px' }}>
        
        {/* Playlist Sidebar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          {podcastList.map((pod) => (
            <div 
              key={pod.id}
              onClick={() => setActivePodcast(pod)}
              style={{
                padding: '15px',
                background: activePodcast?.id === pod.id ? '#e6f7ff' : '#f8f9fa',
                border: activePodcast?.id === pod.id ? '1px solid #1890ff' : '1px solid #eee',
                borderRadius: '12px',
                cursor: 'pointer',
                transition: 'all 0.2s',
                boxShadow: activePodcast?.id === pod.id ? '0 4px 12px rgba(24,144,255,0.15)' : 'none'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontWeight: '600', color: '#333' }}>{pod.topic}</span>
              </div>
              <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '5px', display: 'flex', gap: '10px' }}>
                <span>{Array.isArray(pod.script) ? pod.script.length : 0} segments</span>
                <span>•</span>
                <span>{Math.round((pod.duration_seconds || 0) / 60)} min</span>
              </div>
            </div>
          ))}
          {podcastList.length === 0 && (
            <div style={{ padding: '20px', textAlign: 'center', color: '#999', background: '#f9f9f9', borderRadius: '12px' }}>
              No podcasts available. Generate one in the Setup.
            </div>
          )}
        </div>

        {/* Player Area */}
        <div style={{ background: '#fff', borderRadius: '16px', padding: '40px', border: '1px solid #eaeaea', boxShadow: '0 8px 30px rgba(0,0,0,0.05)' }}>
          {activePodcast ? (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '25px' }}>
                 <div style={{ width: '50px', height: '50px', background: '#007bff', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: '0.8rem', fontWeight: 'bold' }}>
                    STUDIO
                 </div>
                 <h3 style={{ margin: 0, fontSize: '1.8rem', fontWeight: '700' }}>{activePodcast.topic}</h3>
              </div>
              
              {/* Audio Player */}
              {activePodcast.audio_path ? (
                <div style={{ margin: '30px 0', padding: '30px', background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)', borderRadius: '16px', textAlign: 'center' }}>
                  <audio 
                    controls 
                    src={`${BASE_URL}/podcasts/${activePodcast.audio_path}`} 
                    style={{ width: '100%', outline: 'none' }}
                  />
                </div>
              ) : (
                <div style={{ margin: '30px 0', padding: '20px', background: '#fff1f0', borderRadius: '12px', color: '#cf1322', textAlign: 'center' }}>
                  Audio generation failed or is in progress.
                </div>
              )}

              {/* Script View */}
              <div style={{ marginTop: '40px' }}>
                <h4 style={{ borderBottom: '2px solid #f0f0f0', paddingBottom: '15px' }}>Transcript</h4>
                <div className="transcript-container" style={{ display: 'flex', flexDirection: 'column', gap: '20px', maxHeight: '500px', overflowY: 'auto', paddingRight: '15px' }}>
                  {Array.isArray(activePodcast.script) ? activePodcast.script.map((seg, idx) => (
                    <div key={idx} style={{ 
                      display: 'flex', gap: '20px', 
                      padding: '15px', borderRadius: '10px',
                      background: seg.speaker === 'Host 1' ? '#f0f7ff' : '#fff0f6'
                    }}>
                      <div style={{ fontWeight: '800', minWidth: '70px', color: seg.speaker === 'Host 1' ? '#0056b3' : '#c41d7f' }}>
                        {seg.speaker}
                      </div>
                      <div style={{ flex: 1, lineHeight: '1.7', color: '#2c3e50' }}>
                        {seg.text}
                      </div>
                    </div>
                  )) : (
                    <p style={{ color: '#999' }}>Transcript not available.</p>
                  )}
                </div>
              </div>

            </div>
          ) : (
            <div style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#999', flexDirection: 'column' }}>
              <p style={{ fontSize: '1.2rem' }}>Select a deep dive to start the experience.</p>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default Podcast;
