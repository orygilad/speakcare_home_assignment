import os
from utils.audio_utils import convert_m4a_to_wav
from pathlib import Path


def convert_all_m4a_to_wav_in_folder(folder_path):
    print(f"Scanning folder: {folder_path}")
    count = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".m4a"):
                input_path = os.path.join(root, file)
                output_path = os.path.splitext(input_path)[0] + ".wav"

                print(f"[{count + 1}] Converting: {input_path} → {output_path}")
                try:
                    convert_m4a_to_wav(input_path, output_path)
                    print("   ✅ Success")
                    count += 1
                except Exception as e:
                    print(f"   ❌ Failed to convert {input_path}: {e}")

    if count == 0:
        print("No .m4a files found.")
    else:
        print(f"Finished: Converted {count} file(s).")

if __name__ == "__main__":
    # Compute correct base path relative to script
    base_path = Path(__file__).resolve().parent.parent  # goes up from `utils/`
    speakers_path = base_path / "data" / "speakers"
    sessions_path = base_path / "data" / "test_audio"
    convert_all_m4a_to_wav_in_folder(speakers_path)
    convert_all_m4a_to_wav_in_folder(sessions_path)
