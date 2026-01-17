def validate_response(response, threshold=0.5):
    if response.get("confidence", 0) < threshold:
        return {
            "status": "LOW_CONFIDENCE",
            "message": "Insufficient policy coverage",
            "suggestion": "Escalate to HR / Legal"
        }
    return response
