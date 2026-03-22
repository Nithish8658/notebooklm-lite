import { useRef, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { sendMessage } from "../api";
import { useGlobalState } from "../GlobalState";

function getMessageText(message) {
  if (typeof message?.text === "string") {
    return message.text;
  }
  if (typeof message?.reply === "string") {
    return message.reply;
  }
  if (typeof message?.content === "string") {
    return message.content;
  }
  if (message?.text == null && message?.reply == null && message?.content == null) {
    return "";
  }
  return String(message?.text ?? message?.reply ?? message?.content ?? "");
}

function Chat() {
  const { user, messages, setMessages, chatInput: input, setChatInput: setInput } = useGlobalState();
  const [loading, setLoading] = useState(false);
  const [complexity, setComplexity] = useState("Undergrad");
  const [tutorMode, setTutorMode] = useState(false);
  const [bypassRag, setBypassRag] = useState(false);
  const messagesEndRef = useRef(null);

  const complexityLevels = ["5-Year-Old", "High School", "Undergrad", "PhD Expert"];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  useEffect(() => {
    if (messages.length > 0) {
      scrollToBottom();
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) {
      return;
    }

    const userMessage = input;
    setMessages(prev => [...prev, { role: "user", text: userMessage }]);
    setInput("");
    setLoading(true);

    try {
      const reply = await sendMessage(
        user.username, 
        user.active_batch_id, 
        userMessage, 
        complexity, 
        tutorMode,
        bypassRag
      );
      setMessages(prev => [...prev, { role: "bot", text: reply }]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { role: "error", text: err.message }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <h2>NotebookLM Lite</h2>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          {/* Direct LLM Toggle */}
          <button
            onClick={() => setBypassRag(!bypassRag)}
            style={{
              padding: '8px 16px',
              borderRadius: '20px',
              border: bypassRag ? '2px solid #6f42c1' : '1px solid #ccc',
              background: bypassRag ? '#f3f0ff' : '#fff',
              color: bypassRag ? '#6f42c1' : '#666',
              fontSize: '0.8rem',
              fontWeight: '700',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.3s',
              boxShadow: bypassRag ? '0 0 15px rgba(111, 66, 193, 0.3)' : 'none'
            }}
          >
            <span style={{ fontSize: '1.1rem' }}>⚡</span> 
            <span>DIRECT LLM</span>
          </button>

          {/* Advanced Tutor Mode Toggle */}
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setTutorMode(!tutorMode)}
              style={{
                padding: '8px 16px',
                borderRadius: '20px',
                border: tutorMode ? '2px solid #28a745' : '1px solid #ccc',
                background: tutorMode ? '#e6ffed' : '#fff',
                color: tutorMode ? '#28a745' : '#666',
                fontSize: '0.8rem',
                fontWeight: '700',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
                boxShadow: tutorMode ? '0 0 15px rgba(40, 167, 69, 0.4)' : 'none',
                transform: tutorMode ? 'scale(1.05)' : 'scale(1)',
                zIndex: 2
              }}
            >
              <span style={{ fontSize: '1.1rem' }}>🎓</span> 
              <span>SOCRATIC TUTOR</span>
            </button>
            <div style={{
              position: 'absolute',
              top: '-10px',
              right: '-5px',
              background: '#ffc107',
              color: '#000',
              fontSize: '0.6rem',
              padding: '2px 6px',
              borderRadius: '10px',
              fontWeight: 'bold',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              pointerEvents: 'none',
              zIndex: 3
            }}>
              ADVANCED
            </div>
          </div>

          {/* Complexity Toggle */}
          <div style={{ display: 'flex', background: '#f0f0f0', borderRadius: '20px', padding: '2px' }}>
            {complexityLevels.map(level => (
              <button
                key={level}
                onClick={() => setComplexity(level)}
                style={{
                  padding: '4px 12px',
                  borderRadius: '18px',
                  border: 'none',
                  fontSize: '0.75rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  background: complexity === level ? '#fff' : 'transparent',
                  boxShadow: complexity === level ? '0 2px 4px rgba(0,0,0,0.1)' : 'none',
                  color: complexity === level ? '#007bff' : '#666',
                  transition: 'all 0.2s'
                }}
              >
                {level}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="messages-list">
        {messages.map((m, i) => {
          // Extract suggested questions if they exist
          const safeText = getMessageText(m);
          let displayText = safeText;
          let suggestedQuestions = [];
          
          if (m.role === 'bot') {
            const sqRegex = /<sq>(.*?)<\/sq>/g;
            const matches = [...safeText.matchAll(sqRegex)];
            suggestedQuestions = matches.map(match => match[1]);
            // Remove tags from the text shown in the markdown
            displayText = safeText.replace(sqRegex, '').trim();
          }

          return (
            <div key={i} className={`message-row ${m.role}`}>
              <div className="message-bubble" style={{ position: 'relative' }}>
                <div className="message-role">{m.role === 'user' ? 'You' : 'Bot'}</div>
                <div className="message-content">
                  {m.role === "bot" ? (
                    <>
                      <ReactMarkdown>
                        {displayText}
                      </ReactMarkdown>
                      
                      {/* Suggested Question Buttons */}
                      {suggestedQuestions.length > 0 && (
                        <div style={{ 
                          display: 'flex', 
                          flexDirection: 'column', 
                          gap: '8px', 
                          marginTop: '15px',
                          paddingTop: '15px',
                          borderTop: '1px solid #eee'
                        }}>
                          <div style={{ fontSize: '0.7rem', color: '#999', fontWeight: 'bold', textTransform: 'uppercase' }}>
                            Follow-up Suggestions:
                          </div>
                          {suggestedQuestions.map((q, idx) => (
                            <button
                              key={idx}
                              className="pulse-button"
                              onClick={() => setInput(q)}
                              style={{
                                padding: '8px 12px',
                                background: '#f8f9fa',
                                border: '1px solid #e0e0e0',
                                borderRadius: '8px',
                                textAlign: 'left',
                                fontSize: '0.85rem',
                                color: '#007bff',
                                cursor: 'pointer',
                                transition: 'all 0.2s',
                                fontWeight: '500'
                              }}
                              onMouseOver={(e) => { e.target.style.background = '#e7f1ff'; e.target.style.borderColor = '#007bff'; }}
                              onMouseOut={(e) => { e.target.style.background = '#f8f9fa'; e.target.style.borderColor = '#e0e0e0'; }}
                            >
                              {q}
                            </button>
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    displayText || ""
                  )}
                </div>
              </div>
            </div>
          );
        })}
        {loading && <div className="message-row bot"><div className="message-bubble typing">...</div></div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Type a message..."
          disabled={loading}
        />
        <button onClick={handleSend} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

export default Chat;
