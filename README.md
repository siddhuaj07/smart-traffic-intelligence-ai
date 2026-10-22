# Smart Traffic Intelligence AI

## Setup with Anaconda Prompt (Windows)

```bat
conda create -n traffic-ai python=3.11 -y
conda activate traffic-ai
cd %USERPROFILE%\Downloads\Smart_Traffic_Intelligence_AI
pip install -r requirements.txt
streamlit run app.py
```

If you extracted the project to a different folder, change the `cd` command to that folder's full path.

## Use
1. The app opens in a browser.
2. Upload an MP4, AVI, MOV, or MKV traffic video.
3. Choose the YOLO model and confidence threshold.
4. Click Analyze traffic.
5. Download CSV, PDF, and annotated video.

The first run may download pretrained YOLO weights and therefore needs internet access.

## Notes
Vehicle counts are based on tracks crossing a horizontal virtual line. Camera angle, occlusion, lighting, and tracking errors can affect accuracy. The application estimates traffic volume; it does not independently establish congestion or speed.
