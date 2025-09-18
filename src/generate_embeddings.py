import json
from pathlib import Path
from deepface import DeepFace

# Paths
REFERENCE_DIR = Path("data/reference")
OUTPUT_FILE = Path("data/output/reference_embeddings.json")

# Make sure output folder exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def generate_reference_embeddings():
    embeddings = {}

    for img_path in REFERENCE_DIR.glob("*.*"):  # jpg, png, etc.
        try:
            # Extract embedding using ArcFace
            obj = DeepFace.represent(
                img_path.as_posix(),
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=True
            )

            # Some backends return list of dicts (if multiple faces found)
            # We only want the first face for each reference image
            if isinstance(obj, list):
                obj = obj[0]

            if isinstance(obj, dict) and "embedding" in obj:
                embeddings[img_path.name] = obj["embedding"]
                print(f"[OK] Processed {img_path.name}")
            else:
                print(f"[FAIL] {img_path.name}: No embedding found in result")

        except Exception as e:
            print(f"[FAIL] {img_path.name}: {e}")

    # Save embeddings to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(embeddings, f)

    print(f"\n✅ Saved embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_reference_embeddings()
