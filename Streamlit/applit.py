import cv2
from ultralytics import YOLO
import time
import streamlit as st

st.markdown(
    """
    <style>
    .stApp {
        max-width: 800px;
        margin: auto;
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        cursor: pointer;
        border-radius: 4px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #3e8e41;
    }
    .stCheckbox>div>div {
        display: flex;
        align-items: center;
    }
    .stImage>div>div {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Initialize YOLO model
model = YOLO('yolov8n-pose.pt')

# Streamlit app title and checkbox to start/stop
st.title("Real-Time Keypoints Detection with YOLOv8")
st.markdown("*Putu Rayno Sebastian*.")
run_app = st.checkbox("Run")

# Placeholder for displaying the webcam frame and missing keypoints info
frame_placeholder = st.empty()
missing_keypoints_placeholder = st.empty()

# Initialize variables for tracking keypoints and timing
total_keypoints = 0
missing_keypoints_sum = 0
frame_count = 0
start_time = time.time()
print_interval = 1 

# Open webcam feed
cap = cv2.VideoCapture(0)

# Main loop to run the app
while run_app:
    hasFrame, frame = cap.read()

    if not hasFrame:
        st.error("Error: Can't receive frame from webcam.")
        break

    # Perform inference with YOLO model
    output = model(frame, save=False, verbose=False)

    if output is not None:
        # Calculate total keypoints detected in the frame
        total_keypoints_in_frame = sum(len(k.data[0]) for k in output[0].keypoints[0])
        total_keypoints += total_keypoints_in_frame

        # Calculate missing keypoints ratio
        missing_keypoints_in_frame = total_keypoints - total_keypoints_in_frame
        missing_keypoints_ratio = missing_keypoints_in_frame / total_keypoints if total_keypoints > 0 else 0
        missing_keypoints_sum += missing_keypoints_ratio
        frame_count += 1

        # Draw keypoints on the frame
        try:
            for idx, kpt in enumerate(output[0].keypoints[0]):
                for k in kpt.data[0][:, 0:2].cpu().detach().numpy():
                    k = [int(x) for x in k]
                    frame = cv2.circle(frame, (k[0], k[1]), 8, (0, 0, 255), thickness=3, lineType=cv2.FILLED)
        except Exception as e:
            st.error(e)
            pass

    # Display the annotated frame
    frame_placeholder.image(frame, channels="BGR")

    # Update missing keypoints info per second
    if time.time() - start_time >= print_interval:
        missing_keypoints_placeholder.text(f"Sum of missing keypoints ratios per second: {missing_keypoints_sum:.4f}")
        start_time = time.time()
        missing_keypoints_sum = 0

    if not run_app:
        break

# Calculate average missing keypoints ratio
average_missing_keypoints_ratio_per_second = missing_keypoints_sum / frame_count if frame_count > 0 else 0

# Display final results
st.write(f"Average missing keypoints ratio per second: {average_missing_keypoints_ratio_per_second:.4f}")
st.write(f"Total keypoints: {total_keypoints}")
st.write(f"Total missing keypoints ratio: {missing_keypoints_sum:.4f}")

# Release webcam feed and clean up OpenCV resources
cap.release()
cv2.destroyAllWindows()
