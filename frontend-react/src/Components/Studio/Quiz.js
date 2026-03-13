import React, { useState } from 'react';
import { useGlobalState } from '../../GlobalState';

function Quiz() {
  const { quizzes } = useGlobalState();
  const [activeQuiz, setActiveQuiz] = useState(null); // The quiz object currently being taken
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState(null);
  const [isAnswerChecked, setIsAnswerChecked] = useState(false);
  const [score, setScore] = useState(0);
  const [quizFinished, setQuizFinished] = useState(false);

  // --- Reset/Start Logic ---
  const startQuiz = (quiz) => {
    setActiveQuiz(quiz);
    setCurrentQuestionIndex(0);
    setScore(0);
    setQuizFinished(false);
    resetQuestionState();
  };

  const resetQuestionState = () => {
    setSelectedOption(null);
    setIsAnswerChecked(false);
  };

  const handleOptionSelect = (optionId) => {
    if (!isAnswerChecked) {
      setSelectedOption(optionId);
    }
  };

  const checkAnswer = () => {
    if (!selectedOption) return;
    
    const currentQ = activeQuiz.questions[currentQuestionIndex];
    if (selectedOption === currentQ.correct_option_id) {
      setScore(prev => prev + 1);
    }
    setIsAnswerChecked(true);
  };

  const nextQuestion = () => {
    if (currentQuestionIndex < activeQuiz.questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
      resetQuestionState();
    } else {
      setQuizFinished(true);
    }
  };

  const exitQuiz = () => {
    setActiveQuiz(null);
  };

  // --- Render: Empty State ---
  if (!quizzes || quizzes.length === 0) {
    return (
      <div className="studio-tool">
        <div className="mindmap-empty">
          <p>No quizzes available.</p>
          <p style={{ fontSize: '0.9rem', marginTop: '10px' }}>
            Go to the <strong>Topic Setter</strong> to generate a quiz from your documents.
          </p>
        </div>
      </div>
    );
  }

  // --- Render: Gallery View (Book Covers) ---
  if (!activeQuiz) {
    return (
      <div className="studio-tool">
        <div className="tool-header">
          <h2>Quiz Library</h2>
          <p>Select a topic to test your knowledge.</p>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '30px', padding: '20px' }}>
          {quizzes.map((quiz) => (
            <div 
              key={quiz.id} 
              onClick={() => startQuiz(quiz)}
              style={{
                height: '280px',
                background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                borderRadius: '4px 12px 12px 4px',
                boxShadow: '5px 5px 15px rgba(0,0,0,0.15)',
                position: 'relative',
                cursor: 'pointer',
                transition: 'transform 0.2s',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                padding: '20px',
                textAlign: 'center',
                color: 'white',
                borderLeft: '12px solid #333' // Spine
              }}
              onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-5px)'}
              onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
            >
              <h3 style={{ fontSize: '1.4rem', fontWeight: 'bold', textShadow: '0 2px 4px rgba(0,0,0,0.2)' }}>
                {quiz.topic}
              </h3>
              <div style={{ position: 'absolute', bottom: '20px', fontSize: '0.9rem', opacity: 0.9 }}>
                {quiz.questions.length} Questions
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // --- Render: Results View ---
  if (quizFinished) {
    const percentage = Math.round((score / activeQuiz.questions.length) * 100);
    return (
      <div className="studio-tool" style={{ textAlign: 'center', padding: '60px' }}>
        <h2>Quiz Completed!</h2>
        <div style={{ fontSize: '4rem', fontWeight: 'bold', color: percentage >= 70 ? '#28a745' : '#dc3545', margin: '20px 0' }}>
          {percentage}%
        </div>
        <p style={{ fontSize: '1.2rem', color: '#555' }}>
          You answered {score} out of {activeQuiz.questions.length} correctly.
        </p>
        <div style={{ marginTop: '40px', display: 'flex', gap: '20px', justifyContent: 'center' }}>
          <button onClick={() => startQuiz(activeQuiz)} className="control-btn secondary">Retake Quiz</button>
          <button onClick={exitQuiz} className="control-btn">Back to Library</button>
        </div>
      </div>
    );
  }

  // --- Render: Active Question ---
  const question = activeQuiz.questions[currentQuestionIndex];

  return (
    <div className="studio-tool">
      <div className="viewer-header">
        <button className="back-btn" onClick={exitQuiz}>← Exit Quiz</button>
        <div className="viewer-title">
          <h2 style={{ fontSize: '1.2rem' }}>{activeQuiz.topic}</h2>
          <span className="card-counter">Question {currentQuestionIndex + 1} / {activeQuiz.questions.length}</span>
        </div>
      </div>

      <div style={{ maxWidth: '800px', margin: '0 auto' }}>
        <h3 style={{ fontSize: '1.4rem', lineHeight: '1.5', marginBottom: '30px' }}>
          {question.question}
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          {question.options.map((opt) => {
            let bgColor = '#fff';
            let borderColor = '#ddd';
            
            if (isAnswerChecked) {
              if (opt.id === question.correct_option_id) {
                bgColor = '#d4edda'; // Green for correct
                borderColor = '#c3e6cb';
              } else if (opt.id === selectedOption && selectedOption !== question.correct_option_id) {
                bgColor = '#f8d7da'; // Red for wrong selection
                borderColor = '#f5c6cb';
              }
            } else if (selectedOption === opt.id) {
              bgColor = '#e7f1ff';
              borderColor = '#007bff';
            }

            return (
              <div 
                key={opt.id}
                onClick={() => handleOptionSelect(opt.id)}
                style={{
                  padding: '15px 20px',
                  border: `2px solid ${borderColor}`,
                  borderRadius: '10px',
                  backgroundColor: bgColor,
                  cursor: isAnswerChecked ? 'default' : 'pointer',
                  transition: 'all 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '15px'
                }}
              >
                <div style={{ 
                  fontWeight: 'bold', 
                  width: '30px', 
                  height: '30px', 
                  borderRadius: '50%', 
                  background: isAnswerChecked && opt.id === question.correct_option_id ? '#28a745' : '#eee',
                  color: isAnswerChecked && opt.id === question.correct_option_id ? 'white' : '#555',
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center'
                }}>
                  {opt.id}
                </div>
                <div style={{ fontSize: '1.05rem' }}>{opt.text}</div>
              </div>
            );
          })}
        </div>

        {/* Action Bar */}
        <div style={{ marginTop: '30px', paddingTop: '20px', borderTop: '1px solid #eee', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          
          {/* Explanation Area */}
          <div style={{ flex: 1, paddingRight: '20px' }}>
            {isAnswerChecked && (
              <div style={{ background: '#f8f9fa', padding: '15px', borderRadius: '8px', borderLeft: '4px solid #007bff' }}>
                <strong>Explanation:</strong> {question.explanation}
              </div>
            )}
          </div>

          <div>
            {!isAnswerChecked ? (
              <button 
                onClick={checkAnswer} 
                className="control-btn"
                disabled={!selectedOption}
                style={{ opacity: !selectedOption ? 0.6 : 1 }}
              >
                Check Answer
              </button>
            ) : (
              <button onClick={nextQuestion} className="control-btn">
                {currentQuestionIndex < activeQuiz.questions.length - 1 ? 'Next Question' : 'Finish Quiz'}
              </button>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}

export default Quiz;