✈ Airline Customer Support Bot  
ASAPP <> PSG Tech Hackathon 2025 – Problem 1 Solution

🚀 Overview

This project is our solution for *Problem Statement 1* of the ASAPP x PSG Tech Hackathon 2025.  
We built an *AI-powered Airline Customer Support Bot* that can:

✅ Automatically classify customer queries into predefined request types  
✅ Learn from mistakes using user feedback (continuous improvement loop)  
✅ Display system performance including accuracy, logs, and classification history  
✅ Provide a real-time conversational interface using *FastAPI + React*

🎯 Problem Statement

Airlines receive an overwhelming number of customer support requests.  
Many of these are common and repetitive (flight status, cancellations, baggage issues, etc.), yet people still prefer speaking to an agent.

🔹 Goal: Automate the *first response layer* of customer interaction  
🔹 Task: Classify the customer's initial message into the right *request type*  
🔹 If correct → proceed; If incorrect → escalate to human agent  
🔹 System must *learn from feedback* and *track performance over time*  

✅ Functional Requirements
| Feature | Description |
|---------|-------------|
| Input | Customer message (string) |
| Output | Predicted request type |
| Request Types | 18+ (Cancel Trip, Flight Status, Missing Bag, etc.) |
| Feedback | User approves or corrects the prediction |
| Continuous Learning | System updates accuracy based on feedback |
| Model Pipeline | Training, evaluation, logging |
| Performance Metrics | Accuracy, total predictions, correct/incorrect count |

⚙ Non-Functional Requirements
Low latency
Reliable logging & monitoring
Scalable architecture
User-friendly UI

🛠 Tech Stack

Frontend
⚛ React.js (CRA)  
🎨 Custom Styled UI (Dark/Light theme)  
📊 Axios for API communication  

Backend
🚀 FastAPI  
🗄 SQLite (for predictions, feedback, logs & analytics)  
📈 Continuous tracking for stats and accuracy  

📂 Project Structure

📁 airline-system-chatbot/
 ├── backend/
 │   ├── main.py          # FastAPI server with classification, feedback & stats routes
 │   ├── model.py         # Placeholder for ML model integration
 │   ├── requirements.txt # Backend dependencies
 │   └── airline_bot_enhanced.db
 ├── frontend/
 │   ├── src/             # React components & UI logic
 │   ├── public/          # HTML template and static files
 │   └── package.json     # Frontend dependencies
 └── README.md            # You're here!

⚡ Setup & Installation

✅ 1. Clone the Repository

sh
git clone https://github.com/<your-username>/<repo-name>.git
cd airline-system-chatbot


✅ 2. Backend Setup

sh
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

Server runs at 👉 http://127.0.0.1:8000

✅ 3. Frontend Setup

sh
cd frontend
npm install
npm start

UI runs at 👉 http://localhost:3000


🧠 Future Enhancements

| Feature                  | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| 🔁 Active Learning       | Train model based on feedback data                         |
| ☁ Cloud Deployment      | FastAPI + React on AWS / Azure / Render                    |
| 🛡 Authentication       | Admin dashboard for monitoring stats & logs                |
| 🗣 Multilingual Support | Handle multiple languages for customers                    |

🏁 Conclusion

This project showcases how AI + FastAPI + React can transform customer service in the airline industry. With real-time feedback and scalable architecture, this bot can evolve into a production-grade assistant.

### ⭐ If you like this project, don’t forget to star the repository!
