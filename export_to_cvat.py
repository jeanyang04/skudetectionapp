import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("export_to_cvat.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

def normalize_label(label):
    """
    Normalize label names to match the format in the XML file.
    Example: "salad green" -> "Salad Green"
    """
    return ' '.join(word.capitalize() for word in label.split())

def export_to_cvat(video_path, detections, frame_count, output_file="annotations.xml"):
    """
    Export detected objects to CVAT for Video 1.1 XML format.
    
    Args:
        video_path (str): Path to the video file.
        detections (list): List of detections per frame.
        frame_count (int): Total number of frames in the video.
        output_file (str): Output XML file name.
    """
    logging.info("Starting export_to_cvat function...")

    # Create root <annotations>
    root = ET.Element("annotations")

    # Add <version>
    version_el = ET.SubElement(root, "version")
    version_el.text = "1.1"

    # Add <meta>
    meta_el = ET.SubElement(root, "meta")
    task_el = ET.SubElement(meta_el, "task")

    # Task details
    task_id = ET.SubElement(task_el, "id")
    task_id.text = "1"
    task_name = ET.SubElement(task_el, "name")
    task_name.text = "video_annotation"
    task_size = ET.SubElement(task_el, "size")
    task_size.text = str(frame_count)
    task_mode = ET.SubElement(task_el, "mode")
    task_mode.text = "interpolation"
    task_overlap = ET.SubElement(task_el, "overlap")
    task_overlap.text = "5"
    task_bugtracker = ET.SubElement(task_el, "bugtracker")
    task_bugtracker.text = ""
    task_created = ET.SubElement(task_el, "created")
    task_created.text = "2025-01-22 07:58:48.956785+00:00"
    task_updated = ET.SubElement(task_el, "updated")
    task_updated.text = "2025-01-22 08:06:33.325158+00:00"
    task_subset = ET.SubElement(task_el, "subset")
    task_subset.text = "Train"
    task_start_frame = ET.SubElement(task_el, "start_frame")
    task_start_frame.text = "0"
    task_stop_frame = ET.SubElement(task_el, "stop_frame")
    task_stop_frame.text = str(frame_count - 1)
    task_frame_filter = ET.SubElement(task_el, "frame_filter")
    task_frame_filter.text = ""

    # Add segments
    segments_el = ET.SubElement(task_el, "segments")
    segment_el = ET.SubElement(segments_el, "segment")
    segment_id = ET.SubElement(segment_el, "id")
    segment_id.text = "1"
    segment_start = ET.SubElement(segment_el, "start")
    segment_start.text = "0"
    segment_stop = ET.SubElement(segment_el, "stop")
    segment_stop.text = str(frame_count - 1)
    segment_url = ET.SubElement(segment_el, "url")
    segment_url.text = "http://35.198.239.246:8080/api/jobs/1"

    # Add owner
    owner_el = ET.SubElement(task_el, "owner")
    owner_username = ET.SubElement(owner_el, "username")
    owner_username.text = "jeanyang"
    owner_email = ET.SubElement(owner_el, "email")
    owner_email.text = "jeanyang.chen@gmail.com"

    # Add assignee (empty)
    assignee_el = ET.SubElement(task_el, "assignee")
    assignee_el.text = ""

    # Add labels
    labels_el = ET.SubElement(task_el, "labels")
    unique_labels = set()
    for frame_detections in detections:
        for detection in frame_detections:
            unique_labels.add(normalize_label(detection["label"]))

    # Predefined labels from CORRECT_FORMAT.xml
    predefined_labels = [
        "Salad Green", "Salad Orange", "Salad Purple", "Salad SkyBlue",
        "Wrap Blue", "Wrap Brown", "Wrap Green", "Wrap Yellow",
        "Onigiri Brown", "Onigiri Red", "Onigiri Blue",
        "SW Pink", "SW Red", "SW Yellow", "SW Blue", "SW Orange",
        "Minisalad Green", "Minisalad Purple", "Minisalad Yellow",
        "SW Peach", "Yogurt Blue", "Yogurt Yellow",
        "Oats Purple", "Oats Blue", "Coca Cola", "100 Plus"
    ]

    for label_name in predefined_labels:
        label_el = ET.SubElement(labels_el, "label")
        name_el = ET.SubElement(label_el, "name")
        name_el.text = label_name
        color_el = ET.SubElement(label_el, "color")
        color_el.text = "#000000"  # Default color, can be customized
        type_el = ET.SubElement(label_el, "type")
        type_el.text = "any"
        attributes_el = ET.SubElement(label_el, "attributes")
        attributes_el.text = ""

    # Add original size
    original_size_el = ET.SubElement(meta_el, "original_size")
    width_el = ET.SubElement(original_size_el, "width")
    width_el.text = "1280"
    height_el = ET.SubElement(original_size_el, "height")
    height_el.text = "720"

    # Add dumped timestamp
    dumped_el = ET.SubElement(meta_el, "dumped")
    dumped_el.text = "2025-01-22 08:06:51.075061+00:00"

    # Add tracks for each detection
    track_id = 0
    for frame_idx, frame_detections in enumerate(detections):
        for detection in frame_detections:
            label = normalize_label(detection["label"])

            # Create <track>
            track_el = ET.SubElement(root, "track")
            track_el.set("id", str(track_id))
            track_el.set("label", label)
            track_el.set("source", "manual")

            # Create <box> for each frame
            box_el = ET.SubElement(track_el, "box")
            box_el.set("frame", str(frame_idx))
            box_el.set("keyframe", "1")
            box_el.set("outside", "0")
            box_el.set("occluded", "0")

            # Add bounding box coordinates
            box_el.set("xtl", str(detection["xtl"]))
            box_el.set("ytl", str(detection["ytl"]))
            box_el.set("xbr", str(detection["xbr"]))
            box_el.set("ybr", str(detection["ybr"]))
            box_el.set("z_order", "0")

            track_id += 1

    # Write the XML to file
    tree = ET.ElementTree(root)
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_str)

    logging.info(f"Annotations exported to {output_file}")


def process_video(video_path, model_path, skip_frames=5, output_file="annotations.xml"):
    """
    Process a video file with a YOLO model and export annotations in CVAT format.
    
    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the YOLO model file.
        skip_frames (int): Number of frames to skip between detections.
        output_file (str): Output XML file name.
    """
    logging.info("Starting process_video function...")

    # Load the YOLO model
    logging.info(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)

    # Open the video file
    logging.info(f"Opening video file: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logging.info(f"Video properties - Frame count: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")

    # Process each frame
    detections_per_frame = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("Reached end of video.")
            break

        # Skip frames if necessary
        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue

        logging.info(f"Processing frame {frame_idx}...")

        # Perform object detection
        results = model(frame)
        frame_detections = []
        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                confidence = float(box.conf)
                xtl, ytl, xbr, ybr = box.xyxy[0].tolist()
                frame_detections.append({
                    "label": cls,
                    "confidence": confidence,
                    "xtl": xtl,
                    "ytl": ytl,
                    "xbr": xbr,
                    "ybr": ybr
                })

        # Store detections for this frame
        detections_per_frame.append(frame_detections)
        frame_idx += 1

    # Release the video capture object
    logging.info("Releasing video capture...")
    cap.release()

    # Export detections to CVAT XML
    logging.info("Exporting detections to CVAT XML...")
    export_to_cvat(video_path, detections_per_frame, frame_count, output_file)
    logging.info("Export to CVAT XML completed.")


if __name__ == "__main__":
    # Input video and model paths
    logging.info("Starting video processing and annotation export...")
    video_path = "test/in/cam1_FM002_FM003_caesar_soba_160125_220125.mp4"  # Replace with your video file path
    model_path = "best.pt"  # Replace with your YOLO model file path

    # Process the video and export annotations
    logging.info("Starting process_video function...")
    process_video(video_path, model_path, skip_frames=5, output_file="test/out/output.xml")
    logging.info("Finished process_video function. Annotations exported.")