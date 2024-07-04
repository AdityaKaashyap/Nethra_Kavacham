import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Define allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.secret_key = 'supersecretkey'  # Change this to a secure key

# Load your trained model
model = load_model('path_to_your_trained_model.h5')  # Update with your model path
img_height, img_width = 224, 224  # Update with your image dimensions

# Function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            processed_image = preprocess_image(filepath)
            
            # Predict the class
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction)
            
            # Define your class labels (replace with your actual labels)
            class_labels = ['class1', 'class2', 'class3', ...]  # Update with your class labels
            
            # Get the predicted class name
            predicted_class_name = class_labels[predicted_class]
            
            return render_template('index.html', prediction=predicted_class_name, filename=filename)
    
    return render_template('index.html')

if __name__ == '_main_':
    app.run(debug=True)