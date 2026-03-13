import React, { useState } from 'react';
import { useGlobalState } from '../../GlobalState';

function Flashcards() {
  const { flashcards } = useGlobalState();
  const [view, setView] = useState('gallery'); // 'gallery' or 'viewer'
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [filteredCards, setFilteredCards] = useState([]);

  if (flashcards.length === 0) {
    return (
      <div className="studio-tool">
        <div className="mindmap-empty">
          <p>No flashcards available in the system.</p>
        </div>
      </div>
    );
  }

  const topics = [...new Set(flashcards.map(card => card.topic))].map(topicName => {
    return {
      name: topicName,
      count: flashcards.filter(c => c.topic === topicName).length,
      preview: flashcards.find(c => c.topic === topicName).question
    };
  });

  const openTopic = (topicName) => {
    const cards = flashcards.filter(card => card.topic === topicName);
    setFilteredCards(cards);
    setSelectedTopic(topicName);
    setCurrentIndex(0);
    setIsFlipped(false);
    setView('viewer');
  };

  const handleNext = () => {
    setIsFlipped(false);
    setTimeout(() => {
      setCurrentIndex((prev) => (prev + 1) % filteredCards.length);
    }, 150);
  };

  const handlePrev = () => {
    setIsFlipped(false);
    setTimeout(() => {
      setCurrentIndex((prev) => (prev - 1 + filteredCards.length) % filteredCards.length);
    }, 150);
  };

  if (view === 'gallery') {
    return (
      <div className="studio-tool">
        <div className="tool-header">
          <h2>Study Flashcards</h2>
          <p>Choose a topic to start practicing.</p>
        </div>
        
        <div className="topics-gallery">
          {topics.map((topic, idx) => (
            <div key={idx} className="topic-card" onClick={() => openTopic(topic.name)}>
              <div className="topic-card-icon">🎴</div>
              <h3>{topic.name}</h3>
              <p className="topic-preview">"{topic.preview}"</p>
              <div className="topic-footer">
                <span>{topic.count} Cards</span>
                <button className="start-btn">Practice →</button>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="studio-tool">
      <div className="viewer-header">
        <button className="back-btn" onClick={() => setView('gallery')}>← Back to Topics</button>
        <div className="viewer-title">
          <h2>{selectedTopic}</h2>
          <span className="card-counter">{currentIndex + 1} / {filteredCards.length}</span>
        </div>
      </div>

      <div className="flashcard-viewer">
        <div 
          className={`flashcard ${isFlipped ? 'flipped' : ''}`} 
          onClick={() => setIsFlipped(!isFlipped)}
        >
          <div className="flashcard-inner">
            <div className="flashcard-front">
              <div className="card-topic">{selectedTopic}</div>
              <div className="card-content">{filteredCards[currentIndex].question}</div>
              <div className="card-hint">Click to flip</div>
            </div>
            <div className="flashcard-back">
              <div className="card-topic">{selectedTopic}</div>
              <div className="card-content">{filteredCards[currentIndex].answer}</div>
              <div className="card-hint">Click to see question</div>
            </div>
          </div>
        </div>

        <div className="flashcard-controls">
          <button onClick={handlePrev} className="control-btn secondary">Previous</button>
          <button onClick={handleNext} className="control-btn">Next</button>
        </div>
      </div>
    </div>
  );
}

export default Flashcards;
