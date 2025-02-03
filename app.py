from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session handling

# Path to store uploaded images
UPLOAD_FOLDER = 'static/img'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Temporary user database (replace with a real database in production)
users = {}

# Load the trained model
model_path = "models/freshness_detector_model.h5"
try:
    model = load_model(model_path)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Function to predict freshness of the image
def predict_freshness(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        
        if prediction[0][0] > 0.5:
            return "Stale"
        else:
            return "Fresh"
    except Exception as e:
        return f"Error processing image: {e}"

@app.route('/')
def index():
    user = session.get('user')  # Get the username from the session
    return render_template('index.html', prediction=None, user=user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        username = request.form['username']  # Get username from the form

        if email in users:
            flash('Email already registered. Please log in.', 'warning')
            return redirect(url_for('login'))

        # Save the user details
        users[email] = {'password': password, 'username': username}
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Validate credentials
        if email in users and users[email]['password'] == password:
            session['user'] = users[email]['username']  # Store username in session
            flash('Login successful!', 'success')
            return redirect(url_for('index'))

        flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove user from the session
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        flash('You must log in to use this feature.', 'warning')
        return redirect(url_for('login'))
    
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded.", user=session.get('user'))
    
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected.", user=session.get('user'))
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return render_template('index.html', prediction="Invalid file type. Please upload a PNG, JPG, or JPEG file.", user=session.get('user'))
    
    # Save the uploaded image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Get prediction
    if model is None:
        prediction = "Model could not be loaded."
    else:
        prediction = predict_freshness(img_path)

    return render_template('index.html', prediction=prediction, uploaded_img=file.filename, user=session.get('user'))

if __name__ == '__main__':
    app.run(debug=True)
