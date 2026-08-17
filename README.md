# VLM Analysis

Uses OpenAI's `gpt-4o` vision model to analyze camera frames for safety-relevant context that pose-based detectors can't see on their own: environmental hazards (spills, clutter, poor lighting), pre-fall risk posture, and an overall safety score. Runs on a timed interval (default every 60s) rather than every frame, since it's an API call.

Part of a small elder-monitoring toolkit, split out here so it can run and be developed independently of [human-detection](../human-detection) and [fall-detection](../fall-detection).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# then edit .env and put your real OPENAI_API_KEY in
```

**Never commit your `.env` file.** It's already in `.gitignore`. If you ever accidentally push a real key, revoke it immediately at https://platform.openai.com/api-keys and issue a new one.

## Run

```bash
python vlm_analysis.py
```

Opens the default webcam, sends a frame to the model every 10 seconds (shortened interval for the standalone demo), and prints/logs the safety analysis. Press `q` to quit.

## Usage as a module

```python
import os
from dotenv import load_dotenv
from vlm_analysis import EnhancedVLMAnalysisSystem

load_dotenv()
system = EnhancedVLMAnalysisSystem(api_key=os.getenv("OPENAI_API_KEY"), analysis_interval=60)
system.start_processing()
result = system.process_frame(frame, human_detected=True)
system.cleanup()
```

## Data

- Analysis results are logged to `enhanced_vlm_analysis_logs.db` (SQLite, created automatically, git-ignored).
- Analyzed frames + metadata are optionally saved under `vlm_frames/` (also git-ignored, since these can contain identifiable images of people).
