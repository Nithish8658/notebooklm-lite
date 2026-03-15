import { Routes, Route, Link } from "react-router-dom";
import Chat from "./Components/Chat";
import Upload from "./Components/Upload";
import SmartIngest from "./Components/SmartIngest";
import S3PresignedIngest from "./Components/S3PresignedIngest";
import Auth from "./Components/Auth";
import { useGlobalState } from "./GlobalState";
import "./App.css";

import Studio from "./pages/Studio";
import TopicSetter from "./Components/Studio/TopicSetter";
import SourceManagement from "./pages/SourceManagement";
import Metrics from "./pages/Metrics";

function App() {
  const { user, logout, switchBatch } = useGlobalState();

  if (!user) {
    return <Auth />;
  }

  return (
    <div className="App">
      <nav style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
        <Link to="/">
          Chat
        </Link>
        <Link to="/sources">
          Sources
        </Link>
        
        {/* Course Switcher Context */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: '#f0f4ff', padding: '4px 12px', borderRadius: '20px', border: '1px solid #d0dfff' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 'bold', color: '#555' }}>COURSE:</span>
          <select 
            value={user.active_batch_id} 
            onChange={(e) => switchBatch(e.target.value)}
            style={{ 
              background: 'transparent', 
              border: 'none', 
              fontWeight: 'bold', 
              color: '#007bff', 
              outline: 'none',
              cursor: 'pointer'
            }}
          >
            {user.enrolled_batches?.map(c => (
              <option key={c.id} value={c.id}>{c.name}</option>
            ))}
          </select>
        </div>

        <Link to="/topic-setter">
          Topic Setter
        </Link>
        <Link to="/studio">
          Studio
        </Link>
        <button onClick={logout} style={{ 
          background: 'none', 
          border: '1px solid #ff4d4f', 
          color: '#ff4d4f', 
          padding: '2px 8px', 
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '0.8rem',
          marginLeft: 'auto'
        }}>
          Logout
        </button>
        <Link to="/metrics" style={{ opacity: 0.7, fontSize: '0.9rem' }}>
          Metrics
        </Link>
      </nav>

      <Routes>
        <Route
          path="/"
          element={
            <div className="main-content">
              <Chat />
              <div className="sidebar-controls">
                <Upload />
                <SmartIngest />
                <S3PresignedIngest />
              </div>
            </div>
          }
        />

        <Route
          path="/sources"
          element={
            <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
              <SourceManagement />
            </div>
          }
        />

        <Route
          path="/topic-setter"
          element={
            <div style={{ flex: 1, padding: '20px', overflow: 'auto' }}>
              <TopicSetter />
            </div>
          }
        />

        <Route
          path="/studio"
          element={<Studio />}
        />

        <Route
          path="/metrics"
          element={<Metrics />}
        />
      </Routes>
    </div>
  );
}

export default App;
