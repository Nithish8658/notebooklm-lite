import React, { createContext, useContext, useState, useEffect } from 'react';
import { fetchFlashcards, fetchQuizzes, fetchPodcasts, switchBatch } from './api';

const GlobalStateContext = createContext();

export const useGlobalState = () => {
  const context = useContext(GlobalStateContext);
  if (!context) {
    throw new Error('useGlobalState must be used within a GlobalStateProvider');
  }
  return context;
};

export const GlobalStateProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem("notebook_user");
    return saved ? JSON.parse(saved) : null;
  });

  const [messages, setMessages] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [lastMetricsFetch, setLastMetricsFetch] = useState(null);
  const [flashcards, setFlashcards] = useState([]);
  const [quizzes, setQuizzes] = useState([]);
  const [podcasts, setPodcasts] = useState([]);
  
  // Sidebar statuses
  const [uploadStatus, setUploadStatus] = useState("");
  const [youtubeStatus, setYoutubeStatus] = useState("");
  const [videoStatus, setVideoStatus] = useState("");
  const [webStatus, setWebStatus] = useState("");

  // Initial Load of Persisted Data
  useEffect(() => {
    if (user && user.username && user.active_batch_id) {
      // Clear old data when context switches to prevent "ghost" data from other courses
      setFlashcards([]);
      setQuizzes([]);
      setPodcasts([]);

      fetchFlashcards(user.username, user.active_batch_id)
        .then(cards => setFlashcards(cards))
        .catch(err => console.log("No flashcards found or backend offline"));
        
      fetchQuizzes(user.username, user.active_batch_id)
        .then(qz => setQuizzes(qz))
        .catch(err => console.log("No quizzes found or backend offline"));

      fetchPodcasts(user.username, user.active_batch_id)
        .then(pods => setPodcasts(pods))
        .catch(err => console.log("No podcasts found or backend offline"));
    }
  }, [user, user?.username, user?.active_batch_id]);

  const logout = () => {
    setUser(null);
    localStorage.removeItem("notebook_user");
  };

  const handleSwitchBatch = async (newBatchId) => {
    try {
      const result = await switchBatch(user.username, newBatchId);
      const updatedUser = { ...user, active_batch_id: result.active_batch_id };
      setUser(updatedUser);
      localStorage.setItem("notebook_user", JSON.stringify(updatedUser));
    } catch (err) {
      console.error("Failed to switch course context:", err);
      alert("Failed to switch course context.");
    }
  };

  const value = {
    user,
    setUser: (u) => {
      setUser(u);
      if (u) localStorage.setItem("notebook_user", JSON.stringify(u));
    },
    logout,
    switchBatch: handleSwitchBatch,
    messages,
    setMessages,
    metrics,
    setMetrics,
    chatInput,
    setChatInput,
    lastMetricsFetch,
    setLastMetricsFetch,
    uploadStatus,
    setUploadStatus,
    youtubeStatus,
    setYoutubeStatus,
    videoStatus,
    setVideoStatus,
    webStatus,
    setWebStatus,
    flashcards,
    setFlashcards,
    quizzes,
    setQuizzes,
    podcasts,
    setPodcasts
  };

  return (
    <GlobalStateContext.Provider value={value}>
      {children}
    </GlobalStateContext.Provider>
  );
};
