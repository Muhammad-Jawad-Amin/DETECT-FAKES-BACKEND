import os
import cv2
import config
import torch
import tempfile
import numpy as np
from time import time
from flask_cors import CORS
from datetime import datetime
from torchvision import transforms
from storage_manager import StorageManager
from flask import Flask, request, jsonify, send_file
from pred_util_fun import (
    pred_vid,
    load_cvit,
    real_or_fake,
    df_face_mtcnn,
)


app = Flask(__name__)
CORS(app)
storage_manager = StorageManager()

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CViT model
model = load_cvit(config.CNN_VIT_WEIGHTS)


def normalize_data():
    """
    Returns a dictionary of normalization transforms for different data types.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        "data": transforms.Compose([transforms.Normalize(mean=mean, std=std)])
    }

    return data_transforms


def preprocess_frame(frames):
    """
    Preprocesses the frames for input to the model.
    """
    # Convert numpy arrays to torch tensors
    df_tensor = torch.tensor(np.array(frames), device=device).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))

    for i in range(len(df_tensor)):
        df_tensor[i] = normalize_data()["data"](df_tensor[i] / 255.0)

    return df_tensor


@app.route("/detect/image", methods=["POST"])
def predict_image():
    if "Image" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["Image"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "Unsupported file format"}), 400

    try:
        start_time = time()  # Start timing

        # Read image file
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frames = [image]  # Single frame for image input

        # Calculate the file size in MB
        file_size_bytes = len(image_bytes)
        file_size_mb = file_size_bytes / (1024 * 1024)  # Convert to megabytes

        # Extract faces from the image using MTCNN
        faces_found, frame_no_boxes, frames_processed = df_face_mtcnn(
            frames, num_frames=1
        )

        if len(faces_found) == 0:
            return jsonify({"error": "No faces detected"}), 400

        # Preprocess the frame and make a prediction
        df_tensor = preprocess_frame(faces_found)
        y, y_val = pred_vid(df_tensor, model)
        label = real_or_fake(y)
        confidence = y_val if y == 0 else 1 - y_val

        # Draw bounding boxes and labels on frames
        for frame_no, bounding_box in frame_no_boxes:
            x1, y1, x2, y2 = [int(v) for v in bounding_box]
            color = (0, 255, 0) if label == "REAL" else (0, 0, 255)

            cv2.rectangle(frames[frame_no], (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frames[frame_no],
                f"{label}: {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        end_time = time()  # End timing

        # Save the processed image to a temporary file
        image_filepath, image_filename = storage_manager.get_image_filepath()
        cv2.imwrite(image_filepath, frames[frame_no])

        # Prepare additional information for the response
        response_data = {
            "imageId": image_filename,
            "confidenceReal": (1 - y_val) * 100,
            "confidenceFake": y_val * 100,
            "predictedLabel": real_or_fake(y),
            "processingTime": round(end_time - start_time, 2),
            "imageSize": round(file_size_mb, 2),
            "detectionDateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "imageUrl": f"/download/image/{image_filename}",
        }

        # Return the response as JSON
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/detect/video", methods=["POST"])
def predict_video():
    if "Video" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["Video"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    if not file.filename.lower().endswith((".avi", ".mp4", ".mov", ".mpeg")):
        return jsonify({"error": "Unsupported file format"}), 400

    # Get the number of frames from the user input; default is 15 if not provided
    num_frames = int(request.form.get("framesToProcess", 15))

    try:
        start_time = time()  # Start timing

        # Calculate the file size in MB
        file.seek(0, os.SEEK_END)  # Move to the end of the file
        file_size_bytes = file.tell()  # Get the current position in bytes
        file_size_mb = file_size_bytes / (1024 * 1024)  # Convert to megabytes
        file.seek(0)  # Reset the file pointer to the beginning for further processing

        # Handle video input
        video_bytes = file.read()

        # Create a named temporary file in memory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name

        # Open the video file
        video = cv2.VideoCapture(temp_file_path)
        frames = []
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps else 0  # Calculate duration in seconds

        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        video.release()

        # Remove the temporary file
        os.unlink(temp_file_path)

        # Extract faces from frames using MTCNN
        faces_found, frame_no_boxes, frames_processed = df_face_mtcnn(
            frames, num_frames=num_frames
        )

        if len(faces_found) == 0:
            return jsonify({"error": "No faces detected"}), 400

        # Preprocess the frames and make a prediction
        df_tensor = preprocess_frame(faces_found)
        y, y_val = pred_vid(df_tensor, model)
        label = real_or_fake(y)
        confidence = y_val if y == 0 else 1 - y_val

        # Draw bounding boxes and labels on frames with faces
        save_video = []
        for frame_no, bounding_box in frame_no_boxes:
            x1, y1, x2, y2 = [int(v) for v in bounding_box]
            color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
            cv2.rectangle(frames[frame_no], (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frames[frame_no],
                f"{label}: {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            save_video.append(frames[frame_no])

        thumbnail_filepath, thumbnail_filename = (
            storage_manager.get_thumbnail_filepath()
        )
        cv2.imwrite(thumbnail_filepath, save_video[0])

        video_filepath, video_filename = storage_manager.get_video_filepath()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            video_filepath,
            fourcc,
            fps,
            (save_video[0].shape[1], save_video[0].shape[0]),
        )

        for save_frame in save_video:
            out.write(save_frame)
        out.release()

        end_time = time()  # Ending Time

        response_data = {
            "videoId": video_filename,
            "confidenceReal": (1 - y_val) * 100,
            "confidenceFake": y_val * 100,
            "predictedLabel": real_or_fake(y),
            "processingTime": round(end_time - start_time, 2),
            "videoSize": round(file_size_mb, 2),
            "processedFrames": frames_processed,
            "totalFrames": frame_count,
            "videoDuration": round(duration, 2),
            "detectionDateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "videoUrl": f"/download/video/{video_filename}",
            "thumbnailUrl": f"/download/thumbnail/{thumbnail_filename}",
        }

        # Return the response as JSON
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/download/image/<filename>", methods=["GET"])
def download_image(filename):
    """Serve the processed image file for download."""
    storage_manager.clear_processed_images()
    original_file_path = os.path.join(config.PROCESSED_IMAGES_DIR, filename)
    return send_file(original_file_path, as_attachment=True, mimetype="image/png")


@app.route("/download/video/<filename>", methods=["GET"])
def download_video(filename):
    """Serve the processed video file for download."""
    storage_manager.clear_processed_videos()
    original_file_path = os.path.join(config.PROCESSED_VIDEOS_DIR, filename)
    return send_file(original_file_path, as_attachment=True, mimetype="video/mp4")


@app.route("/download/thumbnail/<filename>", methods=["GET"])
def download_thumbnail(filename):
    """Serve the video thumbnail file for download."""
    storage_manager.clear_processed_thumbnails()
    original_file_path = os.path.join(config.VIDEO_THUMBNAILS_DIR, filename)
    return send_file(original_file_path, as_attachment=True, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    # app.run(port=80, debug=True)

# Change the host to '0.0.0.0' to make the app accessible from other devices
# app.run(host="0.0.0.0", port=5000, debug=True)
# ngrok http --domain=bursting-intimate-weasel.ngrok-free.app 80
# waitress-serve --port=5000 app:app
