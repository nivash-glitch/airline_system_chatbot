import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [message, setMessage] = useState('');
  const [conversation, setConversation] = useState([]);
  const [isWaiting, setIsWaiting] = useState(false);
  const [stats, setStats] = useState(null);
  const [darkMode, setDarkMode] = useState(true);
  const [showLogs, setShowLogs] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [logs, setLogs] = useState([]);
  const [history, setHistory] = useState([]);

  const API_URL = 'http://127.0.0.1:8000';

  // Fetch stats
  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error("Error fetching stats:", error);
    }
  };

  // Fetch logs
  const fetchLogs = async () => {
    try {
      const response = await axios.get(`${API_URL}/logs`);
      setLogs(response.data.logs);
      setShowLogs(true);
    } catch (error) {
      console.error("Error fetching logs:", error);
    }
  };

  // Fetch history
  const fetchHistory = async () => {
    try {
      const response = await axios.get(`${API_URL}/history`);
      setHistory(response.data.history);
      setShowHistory(true);
    } catch (error) {
      console.error("Error fetching history:", error);
    }
  };

  // Handle sending a message
  const handleSend = async () => {
    if (!message.trim()) return;

    const userMessage = { sender: 'user', text: message };
    setConversation(prev => [...prev, userMessage]);
    setMessage('');
    setIsWaiting(true);

    try {
      const response = await axios.post(`${API_URL}/classify`, { message });
      const { prediction_id, predicted_request_type, confidence } = response.data;

      const botMessage = {
        sender: 'bot',
        text: `It looks like you're asking about: **${predicted_request_type}**`,
        subtext: `Confidence: ${(confidence * 100).toFixed(0)}%`,
        prediction_id: prediction_id,
        showFeedback: true
      };
      setConversation(prev => [...prev, botMessage]);

    } catch (error) {
      console.error("Error classifying message:", error);
      const errorMessage = { sender: 'bot', text: "Sorry, something went wrong. Please try again." };
      setConversation(prev => [...prev, errorMessage]);
    } finally {
      setIsWaiting(false);
    }
  };

  // Handle feedback
  const handleFeedback = async (prediction_id, is_correct) => {
    try {
      await axios.post(`${API_URL}/feedback`, { prediction_id, is_correct });
      
      setConversation(prev => prev.map(msg => 
        msg.prediction_id === prediction_id ? { ...msg, showFeedback: false, feedback_given: true } : msg
      ));
      
      const feedbackResponse = { 
        sender: 'bot', 
        text: is_correct ? "Great! I'm glad I got that right! 🎉" : "Thanks for the feedback. I'll learn from this! 📚" 
      };
      setConversation(prev => [...prev, feedbackResponse]);

      fetchStats();

    } catch (error) {
      console.error("Error sending feedback:", error);
    }
  };

  // Clear conversation
  const clearConversation = () => {
    setConversation([]);
  };

  // Initial load
  useEffect(() => {
    fetchStats();
  }, []);

  return (
    <div className={`App ${darkMode ? 'dark' : 'light'}`}>
      {/* Header */}
      <header className="App-header">
        <div className="header-content">
          <h1>✈️ Airline Customer Bot</h1>
          <div className="header-actions">
            <button className="theme-toggle" onClick={() => setDarkMode(!darkMode)}>
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
        {stats && stats.overall && (
          <div className="stats-bar">
            <div className="stat-item">
              <span className="stat-label">Total Predictions</span>
              <span className="stat-value">{stats.overall.total_predictions}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Accuracy</span>
              <span className="stat-value">{stats.overall.overall_accuracy}%</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Correct</span>
              <span className="stat-value">{stats.overall.total_correct}</span>
            </div>
          </div>
        )}
      </header>

      {/* Action Bar */}
      <div className="action-bar">
        <button onClick={fetchLogs} className="action-btn">
          📋 View Logs
        </button>
        <button onClick={fetchHistory} className="action-btn">
          🕒 History
        </button>
        <button onClick={clearConversation} className="action-btn danger">
          🗑️ Clear Chat
        </button>
      </div>

      {/* Chat Window */}
      <div className="chat-window">
        {conversation.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <h3>Welcome to Airline Customer Support!</h3>
            <p>Ask me anything about flights, baggage, cancellations, and more.</p>
          </div>
        ) : (
          conversation.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <div className="message-content">
                <p>{msg.text}</p>
                {msg.subtext && <span className="subtext">{msg.subtext}</span>}
              </div>
              {msg.showFeedback && (
                <div className="feedback-buttons">
                  <button className="feedback-btn correct" onClick={() => handleFeedback(msg.prediction_id, true)}>
                    ✔️ Correct
                  </button>
                  <button className="feedback-btn incorrect" onClick={() => handleFeedback(msg.prediction_id, false)}>
                    ✖️ Incorrect
                  </button>
                </div>
              )}
            </div>
          ))
        )}
        {isWaiting && (
          <div className="message bot typing">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="input-area">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Type your question here..."
          disabled={isWaiting}
        />
        <button onClick={handleSend} disabled={isWaiting || !message.trim()}>
          {isWaiting ? '⏳' : '📨 Send'}
        </button>
      </div>

      {/* Logs Modal */}
      {showLogs && (
        <div className="modal-overlay" onClick={() => setShowLogs(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>📋 System Logs</h2>
              <button className="close-btn" onClick={() => setShowLogs(false)}>✖️</button>
            </div>
            <div className="modal-content">
              {logs.length === 0 ? (
                <p className="empty-message">No logs available</p>
              ) : (
                <table className="logs-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Type</th>
                      <th>Severity</th>
                      <th>Message</th>
                    </tr>
                  </thead>
                  <tbody>
                    {logs.map((log, index) => (
                      <tr key={index} className={`severity-${log.severity.toLowerCase()}`}>
                        <td>{new Date(log.timestamp).toLocaleString()}</td>
                        <td><span className="badge">{log.log_type}</span></td>
                        <td><span className={`severity ${log.severity.toLowerCase()}`}>{log.severity}</span></td>
                        <td>{log.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}

      {/* History Modal */}
      {showHistory && (
        <div className="modal-overlay" onClick={() => setShowHistory(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>🕒 Prediction History</h2>
              <button className="close-btn" onClick={() => setShowHistory(false)}>✖️</button>
            </div>
            <div className="modal-content">
              {history.length === 0 ? (
                <p className="empty-message">No history available</p>
              ) : (
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Message</th>
                      <th>Prediction</th>
                      <th>Confidence</th>
                      <th>Feedback</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((item, index) => (
                      <tr key={index}>
                        <td>{new Date(item.timestamp).toLocaleString()}</td>
                        <td className="message-cell">{item.message}</td>
                        <td><span className="badge prediction">{item.predicted_request_type}</span></td>
                        <td>{(item.confidence_score * 100).toFixed(0)}%</td>
                        <td>
                          {item.is_correct === null ? (
                            <span className="feedback-status pending">Pending</span>
                          ) : item.is_correct ? (
                            <span className="feedback-status correct">✔️ Correct</span>
                          ) : (
                            <span className="feedback-status incorrect">✖️ Incorrect</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
