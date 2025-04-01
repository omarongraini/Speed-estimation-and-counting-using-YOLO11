import cv2
from tracker3 import ObjectCounter  # Importing ObjectCounter from tracker.py

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Check for mouse movement
        point = [x, y]
        print(f"Mouse moved to: {point}")

# Open the video file
cap = cv2.VideoCapture(r'test_video.mp4')

# Define region points for counting
region_points = [(135, 208), (676, 203)]

# Initialize the object counter
counter = ObjectCounter(
    region=region_points,  # Pass region points
    model=r"yolo11s.pt",  # Model for object counting
    classes=[2, 5, 7],  # Detect only person class
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=2,  # Adjust line width for display
)

# Create a named window and set the mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Get video properties for saving the output

frame_width, frame_height, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output
output_file = r'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files

resize_video = (1020, 500)
if resize_video:
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1020, 500))
else:
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if the video ends

    # Resize the frame (optional, adjust as needed)
    frame = cv2.resize(frame, resize_video)

    # Process the frame with the object counter
    processed_frame = counter.count(frame)

    # Write the processed frame to the output video file
    out.write(processed_frame)

    # Show the frame
    cv2.imshow("RGB", processed_frame)
    if cv2.waitKey(1) == 27:  # Press 'Esc' to quit
        break

# Release the video capture and writer objects, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_file}")