import torch
import pickle
from torch.nn.functional import cosine_similarity, softmax
from collections import defaultdict
import config
import numpy as np
import utils.statistics_utils as su
import utils.general_utils as utils
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--speaker_reference_folder", type=str,  default = "speakers/")
parser.add_argument("--session", type=str, default="Treatment")
parser.add_argument("--out", type=str, default="results")

args = parser.parse_args()
DATA_DIR = config.PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / args.out
SPEAKER_AUDIO_DIR =  DATA_DIR / args.speaker_reference_folder
session = args.session
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("🔁 Loading reference embeddings...")
reference_embeddings = torch.load(config.EMBEDDING_DIR / "reference_embeddings.pt")  # [N, D]
with open(config.EMBEDDING_DIR / "reference_labels.pkl", "rb") as f:
    reference_labels = pickle.load(f)

# Group reference segment indices by speaker
speaker_to_indices = defaultdict(list)
for idx, label in enumerate(reference_labels):
    speaker = utils.get_speaker_name(label)
    speaker_to_indices[speaker].append(idx)

speakers = sorted(speaker_to_indices.keys())  # consistent ordering
print(f"✅ Found {len(speakers)} speakers in reference set.")

# Load test embeddings
print(f"🔁 Loading test embeddings for: {session}")
test_embeddings = torch.load(config.EMBEDDING_DIR / f"{session}_embeddings.pt")  # [M, D]
all_speaker_probs , filtered_probs , predicted_speaker_values , certainty_values = utils.predict_speaker_per_test_segment(test_embeddings, reference_embeddings , speakers, speaker_to_indices)
# Save and analyze results
all_speaker_probs_tensor = torch.stack(all_speaker_probs)  # [M, S]
if filtered_probs:
    filtered_tensor = torch.stack(filtered_probs)  # [K, S], K = num filtered segments
    mean_probs = filtered_tensor.mean(dim=0)           # ✅ average over segments
    torch.save(mean_probs, config.DATA_DIR / args.out / f"{session}_speaker_probs.pt")
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