print("🚀🚀🚀 LOADING ROUTES.PY — NEW VERSION 🚀🚀🚀")

import json
from flask import Blueprint, request, jsonify
from utils import process_user_response  # Import your logic from utils.py

webhook_bp = Blueprint("webhook", __name__)

@webhook_bp.route("/process-input", methods=["POST", "GET"])
def webhook():
    print(f"📥 Debug: Received {request.method} request at {request.path}")

    # Handle GET requests (for simple checks)
    if request.method == "GET":
        return jsonify({"message": "Webhook is active!"})

    try:
        # Force JSON parsing; return error if no JSON found
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Missing JSON payload"}), 400

        print("🔍 Debug: Received User Input:", data)

        # Check for the 'user_response' key in the JSON
        user_input = data.get("user_inputs")
        if user_input is None:
            return jsonify({"error": "Missing 'user_inputs' key in JSON payload"}), 400

        # Debug: Show what we're about to process
        print(f"🚀 Calling process_user_response with: {user_input}")

        # Call your function from utils.py
        processed_result = process_user_response(user_input)

        # Debug: Show the processed result
        print(f"✅ Processed result: {processed_result}")

        # Ensure "top_ibm_courses" exists before proceeding
        if "top_ibm_courses" not in processed_result:
            return jsonify({"error": "No course recommendations found"}), 400        

        # Build the entire response as one string using <br> for line breaks
        combined_text = (
            "📚 Based on your information, here are some course recommendations!<br><br>"
        )

        # Sort courses by name if desired
        courses = sorted(processed_result["top_ibm_courses"], key=lambda c: c.get("Course_Name", ""))

        for course in courses:
            course_name = course.get("Course_Name", "")
            rating = course.get("Rating", "")
            pct = course.get("High_Rated_Percentage", "")
            url = course.get("Course_URL", "")

            combined_text += (
                f"🎓 Course: {course_name}<br>"
                f"⭐ Rating: {rating} ({pct} of users highly rated this course)<br>"
                f"🔗 [Course Link]({url})<br><br>"
            )

        # Return just one JSON property: "message" with a string
        return jsonify({"message": combined_text})

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return jsonify({"error": "Invalid request data", "message": str(e)}), 400
    
# ✅ New Route for /your-endpoint
@webhook_bp.route("/your-endpoint", methods=["GET"])
def your_endpoint():
    return jsonify({"message": "This is another endpoint!"})