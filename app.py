import os
from datetime import datetime
from collections import defaultdict

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import subprocess
import imageio_ffmpeg
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(ROOT, "uploads")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

VEHICLES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

st.set_page_config(page_title="Smart Traffic Intelligence AI", page_icon="🚦", layout="wide")
st.title("🚦 Smart Traffic Intelligence AI")
st.caption("AI-powered vehicle detection, tracking, counting, and traffic reporting")

def build_pdf(path, counts, duration_seconds, flow, created):
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Smart Traffic Intelligence AI", styles["Title"]),
        Spacer(1, 8),
        Paragraph("Traffic Analysis Report", styles["Heading2"]),
        Spacer(1, 8),
        Paragraph(f"Generated: {created}", styles["Normal"]),
        Paragraph(f"Video duration: {duration_seconds:.2f} seconds", styles["Normal"]),
        Paragraph(f"Average counted traffic flow: {flow:.2f} vehicles/minute", styles["Normal"]),
        Spacer(1, 16),
    ]
    rows = [["Vehicle type", "Count"]]
    for name in VEHICLES.values():
        rows.append([name, int(counts[name])])
    rows.append(["TOTAL", int(sum(counts.values()))])
    table = Table(rows, colWidths=[280, 100], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#D9EAF7")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([table, Spacer(1, 16),
        Paragraph("Interpretation and limitations", styles["Heading3"]),
        Paragraph(
            "Counts are based on tracked vehicles crossing the configured horizontal line. "
            "Occlusion, camera angle, poor lighting, and tracking errors may affect accuracy. "
            "Traffic flow is total counted vehicles divided by video duration in minutes. "
            "This report does not independently determine congestion or road speed.",
            styles["BodyText"]
        )])
    SimpleDocTemplate(path, pagesize=A4).build(story)

uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])
model_name = st.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt"])
confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
line_percent = st.slider("Counting line position (% of video height)", 20, 80, 50, 5)

if uploaded:
    source_name = os.path.basename(uploaded.name)
    input_path = os.path.join(UPLOADS, source_name)
    with open(input_path, "wb") as file:
        file.write(uploaded.getbuffer())
    st.video(uploaded)

    if st.button("Analyze traffic", type="primary"):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Cannot read this video. Try a standard MP4 file.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frames_total <= 0 or width <= 0 or height <= 0:
            cap.release()
            st.error("Video properties are invalid.")
            st.stop()

        duration = frames_total / fps
        line_y = int(height * line_percent / 100)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_out = os.path.join(OUTPUTS, f"processed_{stamp}.mp4")
        csv_out = os.path.join(OUTPUTS, f"traffic_report_{stamp}.csv")
        pdf_out = os.path.join(OUTPUTS, f"traffic_report_{stamp}.pdf")

        writer = cv2.VideoWriter(
            video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            st.error("Cannot create the output video on this system.")
            st.stop()

        counts = defaultdict(int)
        last_y = {}
        counted_ids = set()
        progress = st.progress(0)
        status = st.empty()
        preview = st.empty()
        model = YOLO(model_name)
        frame_number = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_number += 1

                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=list(VEHICLES.keys()),
                    conf=confidence,
                    verbose=False
                )[0]

                annotated = frame.copy()
                cv2.line(annotated, (0, line_y), (width, line_y), (0, 255, 255), 2)

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()

                    for coords, track_id, class_id in zip(boxes, ids, classes):
                        name = VEHICLES.get(class_id)
                        if name is None:
                            continue
                        x1, y1, x2, y2 = map(int, coords)
                        center_y = (y1 + y2) // 2

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                        cv2.putText(
                            annotated, f"{name} ID:{track_id}",
                            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 220, 0), 2
                        )

                        if track_id in last_y:
                            old_y = last_y[track_id]
                            crossed = (
                                old_y < line_y <= center_y
                                or old_y > line_y >= center_y
                            )
                            if crossed and track_id not in counted_ids:
                                counts[name] += 1
                                counted_ids.add(track_id)
                        last_y[track_id] = center_y

                total = sum(counts.values())
                cv2.putText(
                    annotated, f"Vehicles counted: {total}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
                )
                writer.write(annotated)

                if frame_number % max(1, int(fps)) == 0:
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                progress.progress(min(frame_number / frames_total, 1.0))
                status.write(f"Processed {frame_number:,} of {int(frames_total):,} frames")

        finally:
            cap.release()
            writer.release()

        minutes = duration / 60
        total = sum(counts.values())
        flow = total / minutes if minutes > 0 else 0
        created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        summary = pd.DataFrame(
            [{"Vehicle Type": name, "Count": int(counts[name])} for name in VEHICLES.values()]
        )
        summary.loc[len(summary)] = ["Total", total]
        summary.to_csv(csv_out, index=False)
        build_pdf(pdf_out, counts, duration, flow, created)

        st.success("Analysis completed!")
        a, b, c = st.columns(3)
        a.metric("Vehicles counted", total)
        b.metric("Video duration", f"{minutes:.2f} min")
        c.metric("Traffic flow", f"{flow:.2f} vehicles/min")
        st.subheader("Vehicle count summary")
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.bar_chart(summary[summary["Vehicle Type"] != "Total"].set_index("Vehicle Type"))

        with open(csv_out, "rb") as f:
            st.download_button("Download CSV report", f, file_name=os.path.basename(csv_out), mime="text/csv")
        with open(pdf_out, "rb") as f:
            st.download_button("Download PDF report", f, file_name=os.path.basename(pdf_out), mime="application/pdf")
        with open(video_out, "rb") as f:
            st.download_button("Download processed video", f, file_name=os.path.basename(video_out), mime="video/mp4")
        st.info(f"Output files are also saved in: {OUTPUTS}")
else:
    st.info("Upload a traffic video to begin.")
    st.markdown(
        "### Steps\n"
        "1. Upload an MP4, AVI, MOV, or MKV video.\n"
        "2. Click **Analyze traffic**.\n"
        "3. Review counts and traffic flow.\n"
        "4. Download the CSV report, PDF report, or processed video."
    )
