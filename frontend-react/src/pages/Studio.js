import React, { useState } from 'react';
import Flashcards from '../Components/Studio/Flashcards';
import Quiz from '../Components/Studio/Quiz';
import Podcast from '../Components/Studio/Podcast';

function Studio() {
  const [activeTool, setActiveTool] = useState('flashcards');

  const renderTool = () => {
    switch (activeTool) {
      case 'flashcards':
        return <Flashcards />;
      case 'quiz':
        return <Quiz />;
      case 'podcast':
        return <Podcast />;
      default:
        return <Flashcards />;
    }
  };

  return (
    <div className="studio-container">
      <div className="studio-sidebar">
        <h2>Studio</h2>
        <button 
          className={activeTool === 'flashcards' ? 'active' : ''} 
          onClick={() => setActiveTool('flashcards')}
        >
          🎴 Flashcards
        </button>
        <button 
          className={activeTool === 'quiz' ? 'active' : ''} 
          onClick={() => setActiveTool('quiz')}
        >
          📝 Quiz
        </button>
        <button 
          className={activeTool === 'podcast' ? 'active' : ''} 
          onClick={() => setActiveTool('podcast')}
        >
          🎙️ Audio Overview
        </button>
      </div>
      <div className="studio-content">
        {renderTool()}
      </div>
    </div>
  );
}

export default Studio;
