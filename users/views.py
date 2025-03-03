import tensorflow as tf
#from tensorflow.keras import backend as K
from django.shortcuts import render, redirect
from .forms import UserRegistrationForm
from django.http import HttpResponse
from .models import User
from django.contrib.auth import authenticate, login
import os
import keras
import numpy as np
from keras import backend as K
import cv2
from django.shortcuts import render
from .ml_model.preprocessing import preprocess_image_from_variable


def home(request):
    return render(request, 'users/homepage.html')

# Registration view
def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserRegistrationForm()
    return render(request, 'users/register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)  # Django authentication
        if user is not None:
            login(request, user)  # Logs in user and creates session
            request.session['user_id'] = user.id  # Explicitly store user ID in session
            print(f"✅ Authentication successful for user: {user}")
            return redirect('verify_signature')
        else:
            print("❌ Authentication failed: Incorrect username or password.")
    
    return render(request, 'users/login.html')




# Load the model once when the server starts
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml_model', 'signature_model_04.keras')
keras.config.enable_unsafe_deserialization()

# Register the custom function
@tf.keras.utils.register_keras_serializable()
def compute_manhattan_distance(tensors):
    """Calculate the Manhattan distance between two tensors."""
    x, y = tensors
    return tf.abs(x - y)

custom_objects={
    "compute_manhattan_distance": compute_manhattan_distance
}
signature_model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
# print(signature_model.input_names)

def verify_signature(request):
    # Check if user is logged in using session
    if 'user_id' not in request.session:
        return redirect('/login/')  # ✅ Redirect correctly

    # Process POST request with uploaded files
    if request.method == 'POST' and 'signature1' in request.FILES and 'signature2' in request.FILES:
        try:
            # Get the uploaded files
            signature1 = request.FILES['signature1']
            signature2 = request.FILES['signature2']

            # Read the images as numpy arrays
            signature1_array = np.frombuffer(signature1.read(), np.uint8)
            signature2_array = np.frombuffer(signature2.read(), np.uint8)

            # Decode the images
            image1 = cv2.imdecode(signature1_array, cv2.IMREAD_COLOR)
            image2 = cv2.imdecode(signature2_array, cv2.IMREAD_COLOR)

            # Validate image decoding
            if image1 is None or image2 is None:
                return render(request, 'users/verify_signature.html', {
                    'error': "Failed to process uploaded images. Please upload valid image files."
                })

            # Preprocess the images
            image1_preprocessed = preprocess_image_from_variable(image1)
            image2_preprocessed = preprocess_image_from_variable(image2)

            # Add batch dimensions
            image1_preprocessed = np.expand_dims(image1_preprocessed, axis=0)  # Shape: (1, 128, 128, 1)
            image2_preprocessed = np.expand_dims(image2_preprocessed, axis=0)  # Shape: (1, 128, 128, 1)

            print(f"Image1 shape: {image1_preprocessed.shape}")
            print(f"Image2 shape: {image2_preprocessed.shape}")
            print(f"Model input shapes: {[input.shape for input in signature_model.inputs]}")

            # Make predictions using the trained model
            prediction = signature_model.predict({"image1": image1_preprocessed, "image2": image2_preprocessed})
            similarity_score = prediction[0][0]

            # Define the threshold for similarity
            threshold = 0.5
            is_forged = similarity_score > threshold

            # Generate the result
            result = "The signature is forged" if is_forged else "The signature is legit"

            return render(request, 'users/verify_signature.html', {
                'result': result,
                'similarity_score': round(similarity_score, 2)  # Include similarity score for better feedback
            })

        except Exception as e:
            # Handle unexpected errors during prediction or preprocessing
            return render(request, 'users/verify_signature.html', {
                'error': f"An error occurred while processing the request: {str(e)}"
            })

    # Render the verification page if no POST data
    return render(request, 'users/verify_signature.html')







