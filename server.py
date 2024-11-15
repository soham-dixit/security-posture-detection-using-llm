from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/threats', methods=['POST'])
def receive_threat():
    # Check if JSON data is received
    if request.is_json:
        data = request.get_json()
        
        # Process the log data as needed (for example, print it)
        print("Received log data:", data)
        
        # Respond with a success message
        return jsonify({"message": "Log received successfully"}), 200
    else:
        return jsonify({"error": "Invalid data format. Expected JSON"}), 400

if __name__ == "__main__":
    # Run Flask server on all IPs at port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
