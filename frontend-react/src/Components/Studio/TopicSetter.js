import React, { useState, useMemo } from 'react';
import { 
  generateFlashcards, generateQuiz, generatePodcast, 
  deleteFlashcards, deleteQuiz, deletePodcast 
} from '../../api';
import { useGlobalState } from '../../GlobalState';
import { useNavigate } from 'react-router-dom';

function TopicSetter() {
  const [topicInput, setTopicInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('flashcards'); // 'flashcards' | 'quiz' | 'podcast'
  
  const { user, flashcards, setFlashcards, quizzes, setQuizzes, podcasts, setPodcasts } = useGlobalState();
  const navigate = useNavigate();

  // Derived list of unique topics
  const activeTopics = useMemo(() => {
    if (mode === 'flashcards') {
      const topicCounts = {};
      flashcards.forEach(card => {
        topicCounts[card.topic] = (topicCounts[card.topic] || 0) + 1;
      });
      return Object.entries(topicCounts).map(([name, count]) => ({ name, count: `${count} Cards` }));
    } else if (mode === 'quiz') {
      return quizzes.map(q => ({ name: q.topic, count: `${q.questions.length} Questions` }));
    } else {
      return podcasts.map(p => ({ name: p.topic, count: `${p.script ? p.script.length : 0} Segments` }));
    }
  }, [flashcards, quizzes, podcasts, mode]);

  const handleGenerate = async () => {
    const topics = topicInput
      .split(',')
      .map(t => t.trim())
      .filter(t => t.length > 0);

    if (topics.length === 0) {
      alert("Please enter at least one topic.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      if (mode === 'flashcards') {
        const newCards = await generateFlashcards(user.user_id, user.active_cohort_id, topics);
        if (newCards.length === 0) {
          setError("No content generated.");
        } else {
          setFlashcards(prev => {
            const existingIds = new Set(prev.map(c => `${c.topic}-${c.question}`));
            const uniqueNewCards = newCards.filter(c => !existingIds.has(`${c.topic}-${c.question}`));
            return [...prev, ...uniqueNewCards];
          });
          setTopicInput('');
          if (window.confirm(`${newCards.length} flashcards generated! Go to Studio?`)) navigate('/studio');
        }
      } else if (mode === 'quiz') {
        let generatedCount = 0;
        for (const topic of topics) {
          const quiz = await generateQuiz(user.user_id, user.active_cohort_id, topic);
          if (quiz) {
            setQuizzes(prev => [...prev, quiz]);
            generatedCount++;
          }
        }
        if (generatedCount > 0) {
          setTopicInput('');
          if (window.confirm(`${generatedCount} quizzes generated! Go to Studio?`)) navigate('/studio');
        } else {
          setError("Failed to generate quizzes.");
        }
      } else {
        // Podcast (Background Job)
        let generatedCount = 0;
        for (const topic of topics) {
          // generatePodcast now polls until 'completed'
          await generatePodcast(user.user_id, user.active_cohort_id, topic);
          generatedCount++;
        }

        if (generatedCount > 0) {
          // Fetch the full updated list from backend
          const { fetchPodcasts } = await import('../../api'); 
          const updated = await fetchPodcasts(user.user_id, user.active_cohort_id);
          setPodcasts(updated);
          
          setTopicInput('');
          if (window.confirm(`${generatedCount} podcasts generated! Go to Studio?`)) navigate('/studio');
        } else {
          setError("Failed to generate podcast.");
        }
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to generate content.");
    } finally {
      setLoading(false);
    }
  };

  const removeTopic = async (topicName) => {
    if (!window.confirm(`Remove ${mode} for "${topicName}"?`)) return;
    
    try {
      if (mode === 'flashcards') {
        await deleteFlashcards(user.user_id, user.active_cohort_id, topicName);
        setFlashcards(prev => prev.filter(c => c.topic !== topicName));
      } else if (mode === 'quiz') {
        await deleteQuiz(user.user_id, user.active_cohort_id, topicName);
        setQuizzes(prev => prev.filter(q => q.topic !== topicName));
      } else {
        await deletePodcast(user.user_id, user.active_cohort_id, topicName);
        setPodcasts(prev => prev.filter(p => p.topic !== topicName));
      }
    } catch (err) {
      console.error(err);
      alert(`Failed to delete ${mode}: ${err.message}`);
    }
  };

  return (
    <div className="studio-tool">
      <div className="tool-header">
        <h2>Study Material Generator</h2>
        <p>Create Flashcards, Quizzes, or Audio Podcasts from your documents.</p>
        
        <div style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
          {['flashcards', 'quiz', 'podcast'].map(m => (
            <button 
              key={m}
              onClick={() => setMode(m)}
              style={{
                padding: '8px 16px',
                borderRadius: '20px',
                border: 'none',
                background: mode === m ? '#007bff' : '#e0e0e0',
                color: mode === m ? 'white' : '#333',
                cursor: 'pointer',
                fontWeight: '600',
                textTransform: 'capitalize'
              }}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px', marginTop: '20px' }}>
        
        {/* Left Side: Input Form */}
        <div>
          <h3>Generate New {mode.charAt(0).toUpperCase() + mode.slice(1)}</h3>
          {loading ? (
            <div className="mindmap-loading" style={{ height: '200px' }}>
              <div className="spinner"></div>
              <p>Generating {mode} content...</p>
              {mode === 'podcast' && <small>This may take a minute (Script + Audio Synth)...</small>}
            </div>
          ) : (
            <div className="setup-form">
              {error && <div className="upload-status error" style={{ marginBottom: '20px' }}>{error}</div>}
              
              <label style={{ display: 'block', marginBottom: '10px', fontWeight: '500', color: '#333' }}>
                Topic {mode !== 'flashcards' && "(One per generation)"}
              </label>
              <textarea
                value={topicInput}
                onChange={(e) => setTopicInput(e.target.value)}
                placeholder="e.g. Roman Empire"
                rows={4}
                style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid #ddd', marginBottom: '20px' }}
              />
              
              <button 
                onClick={handleGenerate} 
                className="control-btn"
                style={{ width: '100%' }}
                disabled={!topicInput.trim()}
              >
                Generate
              </button>
            </div>
          )}
        </div>

        {/* Right Side: Management List */}
        <div>
          <h3>Active {mode.charAt(0).toUpperCase() + mode.slice(1)}s</h3>
          {activeTopics.length === 0 ? (
            <div style={{ padding: '40px', textAlign: 'center', background: '#f9f9f9', borderRadius: '12px', color: '#999', border: '2px dashed #eee' }}>
              No {mode} generated yet.
            </div>
          ) : (
            <div className="active-topics-list" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {activeTopics.map((topic, idx) => (
                <div key={idx} style={{ 
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center', 
                  padding: '15px 20px', background: '#fff', border: '1px solid #eaeaea', 
                  borderRadius: '10px', boxShadow: '0 2px 4px rgba(0,0,0,0.02)'
                }}>
                  <div>
                    <div style={{ fontWeight: '600', color: '#333' }}>{topic.name}</div>
                    <div style={{ fontSize: '0.8rem', color: '#888' }}>{topic.count}</div>
                  </div>
                  <button 
                    onClick={() => removeTopic(topic.name)}
                    style={{ background: '#fff1f0', color: '#f5222d', border: '1px solid #ffa39e', padding: '5px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '0.85rem' }}
                  >
                    Remove
                  </button>
                </div>
              ))}
              <div style={{ marginTop: '10px', textAlign: 'right' }}>
                <button onClick={() => navigate('/studio')} className="control-btn secondary" style={{ fontSize: '0.85rem' }}>
                  Go to Studio →
                </button>
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default TopicSetter;