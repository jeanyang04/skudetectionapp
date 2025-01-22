import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ultralytics import YOLO

def export_to_cvat(video_path, detections, frame_count, output_file="annotations.xml"):
    """
    Export detected objects to CVAT for Video 1.1 XML format.
    
    Args:
        video_path (str): Path to the video file.
        detections (list): List of detections per frame.
        frame_count (int): Total number of frames in the video.
        output_file (str): Output XML file name.
    """
    # Create the root element
    annotations = ET.Element("annotations")
    version = ET.SubElement(annotations, "version")
    version.text = "1.1"

    # Create the meta section
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    task_id = ET.SubElement(task, "id")
    task_id.text = "1"
    task_name = ET.SubElement(task, "name")
    task_name.text = "video_annotation"
    task_size = ET.SubElement(task, "size")
    task_size.text = str(frame_count)
    task_mode = ET.SubElement(task, "mode")
    task_mode.text = "interpolation"

    # Add labels (extracted from detections)
    labels = ET.SubElement(task, "labels")
    unique_labels = set()
    for frame_detections in detections:
        for detection in frame_detections:
            unique_labels.add(detection["label"])
    for label_name in unique_labels:
        label = ET.SubElement(labels, "label")
        name = ET.SubElement(label, "name")
        name.text = label_name

    # Add tracks for each detected object
    track_id = 1
    for frame_idx, frame_detections in enumerate(detections):
        for detection in frame_detections:
            label = detection["label"]
            box = detection["box"]  # Bounding box in (x_min, y_min, x_max, y_max) format

            # Find or create the track for this object
            track = None
            for existing_track in annotations.findall("track"):
                if existing_track.attrib["label"] == label:
                    track = existing_track
                    break
            if track is None:
                track = ET.SubElement(annotations, "track", id=str(track_id), label=label)
                track_id += 1

            # Add the bounding box for this frame
            ET.SubElement(
                track,
                "box",
                frame=str(frame_idx),
                xtl=str(box[0]),
                ytl=str(box[1]),
                xbr=str(box[2]),
                ybr=str(box[3]),
                occluded="0",
            )

    # Write the XML to a file
    xml_str = minidom.parseString(ET.tostring(annotations)).toprettyxml(indent="  ")
    with open(output_file, "w") as f:
        f.write(xml_str)

    print(f"Annotations exported to {output_file}")


def process_video(video_path, model_path, skip_frames=5, output_file="annotations.xml"):
    """
    Process a video file with a YOLO model and export annotations in CVAT format.
    
    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the YOLO model file.
        skip_frames (int): Number of frames to skip between detections.
        output_file (str): Output XML file name.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing video: {video_path}")
    print(f"Frame count: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")

    # Process each frame
    detections_per_frame = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if necessary
        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue

        # Perform object detection
        results = model(frame)
        frame_detections = []
        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]  # Class label
                confidence = float(box.conf)  # Confidence score
                bbox = box.xyxy[0].tolist()  # Bounding box in (x_min, y_min, x_max, y_max) format
                frame_detections.append({"label": cls, "confidence": confidence, "box": bbox})

        # Store detections for this frame
        detections_per_frame.append(frame_detections)
        frame_idx += 1

    print("Releasing video capture...")
    cap.release()
    print("Video capture released.")

    print("Exporting detections to CVAT XML...")
    export_to_cvat(video_path, detections_per_frame, frame_count, output_file)
    print("Export to CVAT XML completed.")


if __name__ == "__main__":
    # Input video and model paths
    print("Processing video and exporting annotations to CVAT XML...")
    video_path = "test/in/cam1_FM002_FM003_caesar_soba_160125_220125.mp4"  # Replace with your video file path
    model_path = "best.pt"  # Replace with your YOLO model file path

    # Process the video and export annotations
    print("Starting process_video function...")
    process_video(video_path, model_path, skip_frames=5, output_file="test/out/output.xml")
    print("Finished process_video function. Annotations exported.")