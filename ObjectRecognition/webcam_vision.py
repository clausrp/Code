import cv2
import ollama
import base64
from PIL import Image
import io
import time
import numpy as np

def frame_to_base64(frame):
    """
    Converts an OpenCV frame (numpy array) to a base64 encoded string.
    """
    # Convert the OpenCV BGR image to RGB for PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create a PIL Image from the numpy array
    pil_img = Image.fromarray(rgb_frame)
    
    # Save the image to a byte stream
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='JPEG') # JPEG is good for compression
    
    # Get the base64 string
    encoded_string = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
    return encoded_string

def describe_webcam_stream(model_name="granite3.2-vision:2b", frames_to_skip=10):
    """
    Captures video from the webcam and sends frames to Ollama for description.

    Args:
        model_name (str): The name of the Ollama vision model to use.
        frames_to_skip (int): Number of frames to skip before sending one to Ollama.
                              Higher value means less frequent API calls.
    """
    cap = cv2.VideoCapture(0) # 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Using Ollama model: {model_name}")
    print("Press 'q' to quit.")

    frame_count = 0
    last_description_time = time.time()
    description_interval = 5 # seconds to wait before getting a new description

    current_description = "Waiting for model to describe..."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1

        # Display the current frame
        # Add the current description text to the frame for display
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, current_description, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam Stream - Press "q" to Quit', frame)

        # Send a frame to Ollama periodically, not every single frame
        if time.time() - last_description_time >= description_interval:
            print("\n--- Sending frame to Ollama for description ---")
            try:
                base64_image = frame_to_base64(frame)
                
                # Ollama's chat function for vision models
                response = ollama.chat(
                    model=model_name,
                    messages=[
                        {
                            'role': 'user',
                            'content': 'Describe what you see in this image in detail:',
                            'images': [base64_image]
                        }
                    ],
                    stream=False # Set to True if you want streaming responses
                )
                
                current_description = response['message']['content']
                print(f"Model Description: {current_description}")

            except Exception as e:
                current_description = f"Error: {e}"
                print(current_description)
            
            last_description_time = time.time() # Reset timer

        # Press 'q' to quit the webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stream stopped.")

if __name__ == "__main__":
    describe_webcam_stream()
    