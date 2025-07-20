import torch
import config
from feature_extractor import FeatureExtractor
from utils.audio_utils import load_audio
import pickle
from pathlib import Path
from utils.signal_processing import low_pass_filter
import argparse
config.EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
MODEL_DIR = BASE_DIR / "models" / "speechbrain_ecapa"
extractor = FeatureExtractor(str(MODEL_DIR))

def process_reference_audio(args):
    SPEAKER_AUDIO_DIR = config.DATA_DIR / args.speaker_reference_folder
    reference_embeddings = []
    reference_labels = []
    try:
        print("DATA_DIR =", config.DATA_DIR)
    except AttributeError:
        print("⚠️ Could not find DATA_DIR in config module.")
    for speaker_file in SPEAKER_AUDIO_DIR.glob("*.wav"):
        if speaker_file.suffix.lower() != ".wav":
            continue
        signal, fs = load_audio(speaker_file)
        filtered_signal = low_pass_filter(signal, fs, cutoff_freq=config.VOICE_MAX_FREQUENCY)
        segments = extractor.segment_and_embed(filtered_signal)
        for i, emb in enumerate(segments):
            reference_embeddings.append(emb)
            reference_labels.append(f"{speaker_file}_seg{i+1}")

    torch.save(torch.stack(reference_embeddings), config.EMBEDDING_DIR / "reference_embeddings.pt")
    with open(config.EMBEDDING_DIR / "reference_labels.pkl", "wb") as f:
        pickle.dump(reference_labels, f)

def process_test_audio(args):
    session = args.session
    for test_file in config.TEST_AUDIO_DIR.glob(session+".wav"):
        signal, fs = load_audio(test_file)
        filtered_signal = low_pass_filter(signal, fs, cutoff_freq=config.VOICE_MAX_FREQUENCY)
        segments = extractor.segment_and_embed(filtered_signal)
        torch.save(torch.stack(segments), config.EMBEDDING_DIR / f"{test_file.stem}_embeddings.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker_reference_folder", type=str, default="speakers/")
    parser.add_argument("--session", type=str, default="Treatment")
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    process_reference_audio(args)
    process_test_audio(args)
    print("✅ Embedding cache saved.")
