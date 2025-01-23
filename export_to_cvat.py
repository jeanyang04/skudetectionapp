import cv2
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/export_to_cvat.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

product_colors = {
    "Salad Green": "#3df53d",
    "Salad Orange": "#ff6a4d",
    "Salad Purple": "#b83df5",
    "Salad SkyBlue": "#33ddff",
    "Wrap Blue": "#34d1b7",
    "Wrap Brown": "#910014",
    "Wrap Green": "#66ff66",
    "Wrap Yellow": "#fafa37",
    "Onigiri Brown": "#b25050",
    "Onigiri Red": "#ff0007",
    "Onigiri Blue": "#000000",
    "SW Pink": "#ff00cc",
    "SW Red": "#ff040f",
    "SW Yellow": "#fafa37",
    "SW Blue": "#3208ff",
    "SW Orange": "#ff6a4d",
    "Minisalad Green": "#24b353",
    "Minisalad Purple": "#b83df5",
    "Minisalad Yellow": "#fafa37",
    "SW Peach": "#ed8a5f",
    "Yogurt Blue": "#3d3df5",
    "Yogurt Yellow": "#fafa37",
    "Oats Purple": "#b83df5",
    "Oats Blue": "#33ddff",
    "Coca Cola": "#f9060e",
    "100 Plus": "#fe9254"
}

def normalize_label(label):
    """
    Normalize label names to match the format in the XML file.
    Example: "salad green" -> "Salad Green"
    """
    return ' '.join(word.capitalize() for word in label.split())

def export_to_cvat(video_path, detections, frame_count, output_file="annotations.xml"):
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    from datetime import datetime
    import logging

    logging.info("Starting export_to_cvat function...")

    # Group consecutive detections of the same label into a single track
    tracks = []
    active_tracks = {}  # label -> index in 'tracks' list

    for frame_idx, frame_detections in enumerate(detections):
        for det in frame_detections:
            raw_label = det["label"]
            label = normalize_label(raw_label)
            xtl, ytl, xbr, ybr = det["xtl"], det["ytl"], det["xbr"], det["ybr"]

            if label in active_tracks:
                track_idx = active_tracks[label]
                last_box_frame = tracks[track_idx]["boxes"][-1]["frame"]
                if last_box_frame == frame_idx - 1:
                    tracks[track_idx]["boxes"].append({
                        "frame": frame_idx,
                        "xtl": xtl,
                        "ytl": ytl,
                        "xbr": xbr,
                        "ybr": ybr
                    })
                else:
                    new_track = {
                        "label": label,
                        "boxes": [{
                            "frame": frame_idx,
                            "xtl": xtl,
                            "ytl": ytl,
                            "xbr": xbr,
                            "ybr": ybr
                        }]
                    }
                    tracks.append(new_track)
                    active_tracks[label] = len(tracks) - 1
            else:
                new_track = {
                    "label": label,
                    "boxes": [{
                        "frame": frame_idx,
                        "xtl": xtl,
                        "ytl": ytl,
                        "xbr": xbr,
                        "ybr": ybr
                    }]
                }
                tracks.append(new_track)
                active_tracks[label] = len(tracks) - 1

    # Build XML
    root = ET.Element("annotations")
    version_el = ET.SubElement(root, "version")
    version_el.text = "1.1"

    meta_el = ET.SubElement(root, "meta")
    task_el = ET.SubElement(meta_el, "task")

    now_str = datetime.utcnow().isoformat()
    ET.SubElement(task_el, "id").text = "1"
    ET.SubElement(task_el, "name").text = "video_annotation"
    ET.SubElement(task_el, "size").text = str(frame_count)
    ET.SubElement(task_el, "mode").text = "interpolation"
    ET.SubElement(task_el, "overlap").text = "5"
    ET.SubElement(task_el, "bugtracker").text = ""
    ET.SubElement(task_el, "created").text = now_str
    ET.SubElement(task_el, "updated").text = now_str
    ET.SubElement(task_el, "subset").text = "Train"
    ET.SubElement(task_el, "start_frame").text = "0"
    ET.SubElement(task_el, "stop_frame").text = str(frame_count - 1)
    ET.SubElement(task_el, "frame_filter").text = ""

    segments_el = ET.SubElement(task_el, "segments")
    segment_el = ET.SubElement(segments_el, "segment")
    ET.SubElement(segment_el, "id").text = "1"
    ET.SubElement(segment_el, "start").text = "0"
    ET.SubElement(segment_el, "stop").text = str(frame_count - 1)
    ET.SubElement(segment_el, "url").text = "http://35.198.239.246:8080/api/jobs/1"

    owner_el = ET.SubElement(task_el, "owner")
    ET.SubElement(owner_el, "username").text = "jeanyang"
    ET.SubElement(owner_el, "email").text = "jeanyang.chen@gmail.com"

    assignee_el = ET.SubElement(task_el, "assignee")
    assignee_el.text = ""

    labels_el = ET.SubElement(task_el, "labels")
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
        color_el.text = product_colors[label_name]
        type_el = ET.SubElement(label_el, "type")
        type_el.text = "any"
        attr_el = ET.SubElement(label_el, "attributes")
        attr_el.text = ""

    original_size_el = ET.SubElement(meta_el, "original_size")
    ET.SubElement(original_size_el, "width").text = "1280"
    ET.SubElement(original_size_el, "height").text = "720"

    dumped_el = ET.SubElement(meta_el, "dumped")
    dumped_el.text = now_str

    # Create <track> elements
    for t_id, track_data in enumerate(tracks):
        track_el = ET.SubElement(root, "track")
        track_el.set("id", str(t_id))
        track_el.set("label", track_data["label"])
        track_el.set("source", "manual")

        for i, box in enumerate(track_data["boxes"]):
            # If it's the last annotation in this track, set outside=1
            outside_val = "1" if i == len(track_data["boxes"]) - 1 else "0"

            box_el = ET.SubElement(track_el, "box")
            box_el.set("frame", str(box["frame"]))
            box_el.set("keyframe", "1")
            box_el.set("outside", outside_val)
            box_el.set("occluded", "0")
            box_el.set("xtl", str(box["xtl"]))
            box_el.set("ytl", str(box["ytl"]))
            box_el.set("xbr", str(box["xbr"]))
            box_el.set("ybr", str(box["ybr"]))
            box_el.set("z_order", "0")

    # Write the XML to file
    tree_str = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(tree_str).toprettyxml(indent="  ")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    logging.info(f"Annotations exported to {output_file}")

def process_video(video_path, model_path, skip_frames=5, output_file="annotations.xml"):
    """
    Process a video file with a YOLO model and export annotations in CVAT format.
    Consecutive detections of the same label become a single track.
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
        results = model(frame)
        frame_detections = []

        # The ultralytics YOLO returns results; each result has .boxes
        # Each box has xyxy, conf, class
        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detection = {
                    "label": cls,
                    "confidence": confidence,
                    "xtl": x1,
                    "ytl": y1,
                    "xbr": x2,
                    "ybr": y2
                }
                frame_detections.append(detection)

        detections_per_frame.append(frame_detections)
        frame_idx += 1

    # Release video capture
    logging.info("Releasing video capture...")
    cap.release()

    # Export detections to CVAT XML
    logging.info("Exporting detections to CVAT XML...")
    export_to_cvat(video_path, detections_per_frame, frame_count, output_file)
    logging.info("Export to CVAT XML completed.")

if __name__ == "__main__":
    logging.info("Starting video processing and annotation export...")
    video_path = "test/in/cam1_FM002_FM003_caesar_soba_160125_220125.mp4"  # Replace with your .mp4 path
    model_path = "best.pt"  # Replace with your YOLO model path

    # Process the video and export annotations
    logging.info("Calling process_video...")
    process_video(video_path, model_path, skip_frames=1, output_file="test/out/output.xml")
    logging.info("Finished process_video function. Annotations exported.")