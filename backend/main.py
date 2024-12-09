from flask import Flask, request
import pandas as pd

app = Flask(__name__)

# Test endpoint for connection
@app.route('/test', methods=['GET'])
def test_connection():
    return 'Connection successful!', 200

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    try:
        # Read CSV file
        df = pd.read_csv(file)
        print("We got the csv file")
        print(df)
        # Process your CSV file here
        
        return 'File uploaded successfully', 200
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)