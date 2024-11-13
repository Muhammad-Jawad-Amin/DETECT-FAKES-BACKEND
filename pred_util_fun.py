import cv2
import torch
import numpy as np
from cvit_model import CViT
from facenet_pytorch import MTCNN


device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize MTCNN for face detection
mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device=device)


def load_cvit(cvit_weight):
    model = CViT(
        image_size=224,
        patch_size=7,
        num_classes=2,
        channels=512,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
    )

    model.to(device)
    checkpoint = torch.load(cvit_weight, map_location=device, weights_only=True)

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    _ = model.eval()

    return model


def df_face_mtcnn(frames, num_frames, threshold=0.97):
    """
    Extracts faces from frames using MTCNN.

    Args:
        frames (list): A list of frames from which to extract faces.
        num_frames (int): Maximum number of frames to process.
        threshold (float): Minimum confidence level for face detection. Defaults to 0.97.

    Returns:
        tuple: A tuple containing detected faces, their coordinates, and the count of frames processed.
    """
    faces_found = []
    frame_no_boxes = []
    frames_processed = 0

    for frame_no in range(len(frames)):
        try:
            bounding_boxes, confidences = mtcnn.detect(frames[frame_no])
            if bounding_boxes is not None and confidences is not None:
                for bounding_box, confidence in zip(bounding_boxes, confidences):
                    if frames_processed < num_frames and confidence >= threshold:
                        x1, y1, x2, y2 = [int(v) for v in bounding_box]
                        face_crop = frames[frame_no][y1:y2, x1:x2]
                        if face_crop.size > 0:
                            resized_face = cv2.resize(face_crop, (224, 224))
                            resized_face_bgr = cv2.cvtColor(
                                resized_face, cv2.COLOR_RGB2BGR
                            )
                            faces_found.append(resized_face_bgr)
                            frame_no_boxes.append((frame_no, bounding_box))
                        frames_processed += 1
        except Exception as e:
            print(f"Error extracting faces: {str(e)}")
            continue

        if frames_processed >= num_frames:
            break

    # Convert faces_found list to a numpy array
    faces_found_array = (
        np.stack(faces_found)
        if faces_found
        else np.zeros((0, 224, 224, 3), dtype=np.uint8)
    )

    return (
        (faces_found_array, frame_no_boxes, frames_processed)
        if frames_processed > 0
        else (np.zeros((0, 224, 224, 3), dtype=np.uint8), [], 0)
    )


def pred_vid(df, model):
    with torch.no_grad():
        return max_prediction_value(torch.sigmoid(model(df).squeeze()))


def max_prediction_value(y_pred):
    """
    Finds the index and value of the maximum prediction value.
    """
    mean_val = torch.mean(y_pred, dim=0)

    if mean_val.numel() == 1:
        mean_val = y_pred

    return (
        torch.argmax(mean_val).item(),
        (
            mean_val[0].item()
            if mean_val[0] > mean_val[1]
            else abs(1 - mean_val[1]).item()
        ),
    )


def real_or_fake(prediction):
    """
    Determines the label from the prediction.
    """
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]
