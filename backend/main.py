import sqlite3
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random

# --- Pydantic Models ---
class ClassifyRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    prediction_id: int
    is_correct: bool
    correct_request_type: str = None

# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = "airline_bot_enhanced.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    return conn

def log_event(log_type: str, message: str, severity: str = "INFO", additional_data: dict = None):
    """Helper function to log events"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO logs (log_type, message, severity, additional_data) VALUES (?, ?, ?, ?)",
            (log_type, message, severity, json.dumps(additional_data) if additional_data else None)
        )
        conn.commit()
    except Exception as e:
        print(f"Logging error: {e}")
    finally:
        conn.close()

# --- Mock Data ---
MOCK_REQUEST_TYPES = [
    "Cancel Trip", "Cancellation Policy", "Carry On Luggage Faq",
    "Change Flight", "Check In Luggage Faq", "Complaints",
    "Damaged Bag", "Discounts", "Fare Check", "Flight Status",
    "Flights Info", "Insurance", "Medical Policy", "Missing Bag",
    "Pet Travel", "Prohibited Items Faq", "Seat Availability", "Sports Music Gear"
]

# --- API Endpoints ---
@app.post("/classify")
async def classify_message(request: ClassifyRequest, req: Request):
    """Classify customer message and store prediction"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Mock prediction (replace with actual ML model later)
    mock_prediction = random.choice(MOCK_REQUEST_TYPES)
    mock_confidence = round(random.uniform(0.7, 0.99), 2)
    
    # Get client IP
    client_ip = req.client.host
    
    try:
        cursor.execute(
            "INSERT INTO predictions (message, predicted_request_type, confidence_score, user_ip) VALUES (?, ?, ?, ?)",
            (request.message, mock_prediction, mock_confidence, client_ip)
        )
        conn.commit()
        prediction_id = cursor.lastrowid
        
        # Log the prediction
        log_event("PREDICTION", f"New prediction made: {mock_prediction}", "INFO", 
                  {"prediction_id": prediction_id, "confidence": mock_confidence})
        
        return {
            "prediction_id": prediction_id,
            "message": request.message,
            "predicted_request_type": mock_prediction,
            "confidence": mock_confidence
        }
    except Exception as e:
        conn.rollback()
        log_event("ERROR", f"Prediction failed: {str(e)}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()

@app.post("/feedback")
def receive_feedback(request: FeedbackRequest):
    """Receive and store user feedback"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO feedback (prediction_id, is_correct, correct_request_type) VALUES (?, ?, ?)",
            (request.prediction_id, request.is_correct, request.correct_request_type)
        )
        conn.commit()
        
        # Log feedback
        log_event("FEEDBACK", f"Feedback received for prediction {request.prediction_id}", "INFO",
                  {"is_correct": request.is_correct})
        
        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        conn.rollback()
        log_event("ERROR", f"Feedback storage failed: {str(e)}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()

@app.get("/stats")
def get_system_stats():
    """Get overall system statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get overall stats
        cursor.execute("SELECT * FROM system_stats")
        stats_row = cursor.fetchone()
        
        if stats_row:
            stats = {
                "total_predictions": stats_row[0] or 0,
                "total_feedback": stats_row[1] or 0,
                "total_correct": stats_row[2] or 0,
                "total_incorrect": stats_row[3] or 0,
                "overall_accuracy": stats_row[4] or 0
            }
        else:
            stats = {
                "total_predictions": 0,
                "total_feedback": 0,
                "total_correct": 0,
                "total_incorrect": 0,
                "overall_accuracy": 0
            }
        
        # Get per-category accuracy
        cursor.execute("SELECT * FROM prediction_accuracy")
        category_stats = []
        for row in cursor.fetchall():
            category_stats.append({
                "request_type": row[0],
                "total_predictions": row[1],
                "correct": row[2],
                "incorrect": row[3],
                "accuracy": row[4]
            })
        
        return {
            "overall": stats,
            "by_category": category_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {e}")
    finally:
        conn.close()

@app.get("/logs")
def get_logs(limit: int = 50):
    """Get recent system logs"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        logs = []
        for row in cursor.fetchall():
            logs.append({
                "id": row[0],
                "log_type": row[1],
                "message": row[2],
                "severity": row[3],
                "timestamp": row[4],
                "additional_data": row[5]
            })
        return {"logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching logs: {e}")
    finally:
        conn.close()
