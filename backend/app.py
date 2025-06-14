from flask import Flask

# Initialize the Flask application
app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    # Run the app on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
