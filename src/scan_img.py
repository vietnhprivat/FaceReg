import json
import pandas as pd
from pathlib import Path
from deepface import DeepFace
import numpy as np
import shutil

# Paths
REFERENCE_EMBEDDINGS = Path("data/output/reference_embeddings.json")
TEST_DATA_DIR = Path("data/test_data")
OUTPUT_DIR = Path("data/output")
MATCHES_DIR = OUTPUT_DIR / "matches"
RESULTS_FILE = OUTPUT_DIR / "results.csv"

# Threshold for cosine similarity (tune this!)
# Smaller distance = closer match. ~0.4–0.6 is typical for ArcFace.
THRESHOLD = 0.45  

# Make sure output dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATCHES_DIR.mkdir(parents=True, exist_ok=True)

def load_reference_embeddings():
    with open(REFERENCE_EMBEDDINGS, "r") as f:
        data = json.load(f)
    return np.array(list(data.values()))

def compare_embeddings(face_embedding, reference_embeddings):
    # cosine similarity distance
    dot = np.dot(reference_embeddings, face_embedding)
    norm_ref = np.linalg.norm(reference_embeddings, axis=1)
    norm_face = np.linalg.norm(face_embedding)
    sims = dot / (norm_ref * norm_face)
    best_score = np.max(sims)
    return best_score

def scan_images():
    reference_embeddings = load_reference_embeddings()
    results = []

    for img_path in TEST_DATA_DIR.glob("*.*"):
        try:
            # Detect faces + get embeddings
            objs = DeepFace.represent(
                img_path.as_posix(),
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )

            if not objs:
                print(f"[NO FACE] {img_path.name}")
                results.append({"file": img_path.name, "match": False, "score": None})
                continue

            # If multiple faces, loop through them
            match_found = False
            best_score = -1

            for obj in objs:
                # Ensure obj is a dictionary before accessing "embedding"
                if isinstance(obj, dict) and "embedding" in obj:
                    face_embedding = np.array(obj["embedding"])
                    score = compare_embeddings(face_embedding, reference_embeddings)
                else:
                    continue

                if score > best_score:
                    best_score = score

                if score > (1 - THRESHOLD):  # similarity above threshold
                    match_found = True
                    # save a copy of matched image
                    shutil.copy(img_path, MATCHES_DIR / img_path.name)
                    break

            results.append({
                "file": img_path.name,
                "match": match_found,
                "score": round(float(best_score), 4)
            })

            print(f"[{'MATCH' if match_found else 'NO MATCH'}] {img_path.name} -> {best_score:.4f}")

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            results.append({"file": img_path.name, "match": False, "score": None})

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\n✅ Results saved to {RESULTS_FILE}")
    print(f"✅ Matches copied to {MATCHES_DIR}")

if __name__ == "__main__":
    scan_images()
