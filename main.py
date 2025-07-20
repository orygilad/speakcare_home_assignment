import os
import torch
from feature_extractor import FeatureExtractor
from utils import audio_utils
import config
from collections import defaultdict
import utils.signal_processing as sp
import numpy as np
import utils.statistics_utils as su
import utils.general_utils as utils
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--speaker_reference_folder", type=str , default = "speakers/")
parser.add_argument("--session", type=str, default="Treatment")
parser.add_argument("--out", type=str, default="results")

args = parser.parse_args()
DATA_DIR = config.PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / args.out
SPEAKER_AUDIO_DIR =  DATA_DIR / args.speaker_reference_folder
session = args.session
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
extractor = FeatureExtractor("models/speechbrain_ecapa")
segment_embeddings = []
segment_labels = []

print(f"Scanning {SPEAKER_AUDIO_DIR} for speaker audio...")

for speaker_file in os.listdir(SPEAKER_AUDIO_DIR):
    if not speaker_file.endswith(".wav"):
        continue
    signal, fs = audio_utils.load_audio(os.path.join(SPEAKER_AUDIO_DIR, speaker_file))
    signal = sp.low_pass_filter(signal, fs, cutoff_freq=config.VOICE_MAX_FREQUENCY)

    total_samples = signal.shape[1]
    segment_len = config.SEGMENT_SECONDS * config.SAMPLE_RATE
    num_segments = total_samples // segment_len

    print(f"{speaker_file}: splitting into {num_segments} segments...")

    for i in range(num_segments):
        start = i * segment_len
        end = start + segment_len
        segment = signal[:, start:end]

        with torch.no_grad():
            embedding = extractor.extract_embedding_from_tensor(segment, config.SAMPLE_RATE)

        segment_embeddings.append(torch.tensor(embedding))
        segment_labels.append(f"{speaker_file}_seg{i+1}")
# Group reference segment indices by speaker
speaker_to_indices = defaultdict(list)
for idx, label in enumerate(segment_labels):
    speaker = utils.get_speaker_name(label)
    speaker_to_indices[speaker].append(idx)

speakers = sorted(speaker_to_indices.keys())  # consistent ordering

reference_embeddings = torch.stack(segment_embeddings)
print("Extracted reference embeddings. extracting test embeddings...")
test_signal, fs = audio_utils.load_audio((config.TEST_AUDIO_DIR/ session).with_suffix(".wav"))
test_signal = sp.low_pass_filter(test_signal, fs, cutoff_freq=config.VOICE_MAX_FREQUENCY)

test_embeddings =extractor.segment_and_embed(test_signal)
all_speaker_probs , filtered_probs , predicted_speaker_values , certainty_values = utils.predict_speaker_per_test_segment(test_embeddings, reference_embeddings , speakers, speaker_to_indices)
# Save and analyze results
all_speaker_probs_tensor = torch.stack(all_speaker_probs)  # [M, S]
torch.save(all_speaker_probs_tensor, config.DATA_DIR / args.out / f"{session}_speaker_probs.pt")
if filtered_probs:
    filtered_tensor = torch.stack(filtered_probs)  # [K, S], K = num filtered segments
    mean_probs = filtered_tensor.mean(dim=0)
    torch.save(mean_probs, config.DATA_DIR / args.out / f"{session}_speaker_probs.pt")
    # ✅ average over segments
    variances = filtered_tensor.var(dim=0)
    print("\n📊 Overall speaker probability after filtering:")
    for i, speaker in enumerate(speakers):
        print(f"  {speaker:15s} → {mean_probs[i].item():.3f} +- {np.sqrt(variances[i].item()):.8f}")
    top_idx , confidence = su.compute_confidence(mean_probs , variances)
    top_speaker = speakers[top_idx]
    torch.save(top_speaker, config.DATA_DIR / args.out / f"{session}_predicted_speaker.pt")
    torch.save(confidence, config.DATA_DIR / args.out / f"{session}_predicted_speaker_confidence.pt")
    print(f"\n🏆 Predicted overall speaker for '{session}': {top_speaker} with confidence:{confidence:.5f} ")
else:
    print("\n⚠️ No segments passed the certainty threshold. Cannot aggregate.")