from pathlib import Path

# Project root = location of config.py
PROJECT_ROOT = Path(__file__).resolve().parent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
SPEAKER_AUDIO_DIR = DATA_DIR / "speakers"
TEST_AUDIO_DIR = DATA_DIR / "test_audio"
EMBEDDING_DIR = DATA_DIR / "embeddings"
MODEL_DIR = PROJECT_ROOT / "models" / "speechbrain_ecapa"
#Test sample name
TEST_NAME = "Treatment"
# Audio config
SEGMENT_SECONDS = 3
SAMPLE_RATE = 16000
VOICE_MAX_FREQUENCY = 4000
#SAMPLE_RATE = 44100
CERTAINTY_THRESHOLD = 0.2 # 0.2 for speechbrain_ecapa
