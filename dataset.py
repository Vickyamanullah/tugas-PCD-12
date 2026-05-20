"""
Dataset Generator for Object Matching System
Generates 5 object classes with reference + test images (rotation, scale, illumination, occlusion)
"""

import cv2
import numpy as np
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

DATASET_DIR = Path("dataset")
IMG_SIZE = (300, 300)
np.random.seed(42)


def add_texture(img, obj_type):
    """Add realistic textures to objects"""
    if obj_type == "book":
        # Book cover with lines and text-like patterns
        cv2.rectangle(img, (10, 10), (290, 290), (30, 80, 150), -1)
        cv2.rectangle(img, (10, 10), (290, 290), (20, 60, 120), 3)
        for y in range(50, 260, 25):
            cv2.line(img, (30, y), (270, y), (200, 220, 255), 1)
        for y in range(50, 260, 25):
            w = np.random.randint(60, 200)
            cv2.rectangle(img, (30, y - 8), (30 + w, y - 2), (180, 200, 240), -1)
        cv2.rectangle(img, (10, 10), (35, 290), (15, 50, 100), -1)
        cv2.rectangle(img, (10, 10), (35, 290), (10, 40, 80), 2)

    elif obj_type == "mug":
        # Cylindrical mug
        cv2.ellipse(img, (150, 80), (80, 20), 0, 0, 360, (180, 100, 50), -1)
        cv2.rectangle(img, (70, 80), (230, 250), (200, 120, 60), -1)
        cv2.ellipse(img, (150, 250), (80, 20), 0, 0, 360, (160, 90, 40), -1)
        # Handle
        pts = np.array([[230, 130], [270, 130], [280, 180], [270, 200], [230, 200]], np.int32)
        cv2.polylines(img, [pts], True, (140, 80, 30), 8)
        # Logo on mug
        cv2.circle(img, (150, 165), 35, (220, 150, 80), -1)
        cv2.circle(img, (150, 165), 35, (160, 90, 30), 2)
        cv2.putText(img, "MUG", (127, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 50, 10), 2)
        # Highlight
        cv2.ellipse(img, (105, 120), (15, 40), -20, 0, 360, (230, 160, 100), -1)

    elif obj_type == "bottle":
        # Water bottle shape
        pts_bottle = np.array([
            [130, 30], [170, 30], [175, 80], [185, 100],
            [190, 260], [110, 260], [115, 100], [125, 80]
        ], np.int32)
        cv2.fillPoly(img, [pts_bottle], (100, 160, 220))
        cv2.polylines(img, [pts_bottle], True, (60, 120, 180), 3)
        # Cap
        cv2.rectangle(img, (125, 10), (175, 35), (80, 80, 200), -1)
        cv2.rectangle(img, (125, 10), (175, 35), (60, 60, 160), 2)
        # Label
        cv2.rectangle(img, (112, 130), (188, 220), (240, 240, 255), -1)
        cv2.rectangle(img, (112, 130), (188, 220), (80, 120, 200), 2)
        cv2.putText(img, "WATER", (118, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 80, 180), 1)
        cv2.putText(img, "500ml", (125, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 100, 160), 1)
        # Highlight
        cv2.line(img, (135, 50), (135, 250), (150, 200, 255), 3)

    elif obj_type == "toy":
        # Toy car / block toy
        # Body
        cv2.rectangle(img, (60, 150), (240, 230), (220, 50, 50), -1)
        cv2.rectangle(img, (60, 150), (240, 230), (160, 30, 30), 3)
        # Roof
        cv2.rectangle(img, (90, 100), (210, 155), (240, 80, 80), -1)
        cv2.rectangle(img, (90, 100), (210, 155), (180, 40, 40), 2)
        # Windows
        cv2.rectangle(img, (98, 108), (148, 148), (150, 200, 240), -1)
        cv2.rectangle(img, (155, 108), (205, 148), (150, 200, 240), -1)
        # Wheels
        for cx, cy in [(100, 230), (200, 230)]:
            cv2.circle(img, (cx, cy), 28, (40, 40, 40), -1)
            cv2.circle(img, (cx, cy), 18, (80, 80, 80), -1)
            cv2.circle(img, (cx, cy), 6, (150, 150, 150), -1)
        # Headlights
        cv2.rectangle(img, (62, 162), (85, 178), (255, 255, 150), -1)
        cv2.rectangle(img, (215, 162), (238, 178), (255, 100, 100), -1)

    elif obj_type == "remote":
        # TV Remote
        pts = np.array([
            [115, 20], [185, 20], [200, 280], [100, 280]
        ], np.int32)
        cv2.fillPoly(img, [pts], (50, 50, 50))
        cv2.polylines(img, [pts], True, (30, 30, 30), 3)
        # Display area
        cv2.rectangle(img, (120, 35), (180, 80), (30, 60, 30), -1)
        cv2.rectangle(img, (120, 35), (180, 80), (0, 100, 0), 2)
        # Buttons - power
        cv2.circle(img, (150, 100), 12, (200, 50, 50), -1)
        cv2.circle(img, (150, 100), 12, (150, 30, 30), 2)
        # Number buttons grid
        colors = [(100, 100, 200), (80, 160, 80), (180, 180, 50), (160, 80, 160)]
        btn_positions = [(128, 135), (150, 135), (172, 135),
                         (128, 160), (150, 160), (172, 160),
                         (128, 185), (150, 185), (172, 185),
                         (128, 210), (150, 210), (172, 210)]
        for i, (bx, by) in enumerate(btn_positions):
            cv2.circle(img, (bx, by), 9, colors[i % 4], -1)
            cv2.putText(img, str(i + 1) if i < 9 else ["*", "0", "#"][i - 9],
                        (bx - 5, by + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1)
        # Navigation pad
        cv2.circle(img, (150, 240), 22, (70, 70, 70), -1)
        cv2.circle(img, (150, 240), 10, (90, 90, 90), -1)

    return img


def draw_object(obj_type, size=IMG_SIZE):
    img = np.ones((*size, 3), dtype=np.uint8) * 240  # Light gray background
    # Add subtle gradient background
    for y in range(size[0]):
        val = int(220 + 30 * y / size[0])
        img[y, :] = [val, val, val]
    add_texture(img, obj_type)
    return img


def apply_variation(img, var_type, intensity=1.0):
    """Apply various degradations to create test images"""
    result = img.copy()
    h, w = result.shape[:2]
    cx, cy = w // 2, h // 2

    if var_type == "rotation":
        angle = 25 * intensity * np.random.choice([-1, 1])
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderValue=(200, 200, 200))

    elif var_type == "scale":
        scale = 0.65 + 0.3 * (1 - intensity) if intensity > 0.5 else 1.3 + 0.2 * intensity
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        result = cv2.warpAffine(result, M, (w, h), borderValue=(200, 200, 200))

    elif var_type == "illumination":
        gamma = max(0.1, 0.4 + 1.2 * (1 - intensity))
        table = np.array([(0 if i == 0 else (i / 255.0) ** gamma * 255) for i in range(256)], dtype=np.uint8)
        result = cv2.LUT(result, table)
        # Add noise
        noise = np.random.normal(0, 15 * intensity, result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif var_type == "occlusion":
        # Add occluding rectangles
        n_occ = int(2 + 2 * intensity)
        for _ in range(n_occ):
            x1 = np.random.randint(0, w - 60)
            y1 = np.random.randint(0, h - 60)
            x2 = x1 + np.random.randint(40, 80)
            y2 = y1 + np.random.randint(40, 80)
            color = tuple(np.random.randint(50, 200, 3).tolist())
            cv2.rectangle(result, (x1, y1), (x2, y2), color, -1)

    elif var_type == "combined":
        # Rotation + scale + slight illumination
        angle = 15 * np.random.choice([-1, 1])
        scale = np.random.uniform(0.75, 1.25)
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        result = cv2.warpAffine(result, M, (w, h), borderValue=(200, 200, 200))
        gamma = np.random.uniform(0.6, 1.4)
        table = np.array([(0 if i == 0 else (i / 255.0) ** gamma * 255) for i in range(256)], dtype=np.uint8)
        result = cv2.LUT(result, table)

    return result


def generate_dataset():
    """Generate all dataset images"""
    objects = ["book", "mug", "bottle", "toy", "remote"]
    variations = [
        ("rotation", [0.5, 1.0, 1.5]),
        ("scale", [0.5, 1.0]),
        ("illumination", [0.5, 1.0, 1.5]),
        ("occlusion", [0.5]),
        ("combined", [1.0]),
    ]

    all_images_info = []

    for obj in objects:
        obj_dir = DATASET_DIR / obj
        obj_dir.mkdir(parents=True, exist_ok=True)

        # Generate reference image
        ref_img = draw_object(obj)
        ref_path = obj_dir / "reference.png"
        cv2.imwrite(str(ref_path), ref_img)
        print(f"  [+] {obj}/reference.png")
        all_images_info.append({"object": obj, "type": "reference", "path": str(ref_path)})

        # Generate test images
        test_idx = 1
        for var_type, intensities in variations:
            for intensity in intensities:
                test_img = apply_variation(ref_img, var_type, intensity)
                filename = f"test_{test_idx:02d}_{var_type}.png"
                test_path = obj_dir / filename
                cv2.imwrite(str(test_path), test_img)
                print(f"  [+] {obj}/{filename}")
                all_images_info.append({
                    "object": obj, "type": "test",
                    "variation": var_type, "intensity": intensity,
                    "path": str(test_path)
                })
                test_idx += 1

    return all_images_info


def create_dataset_preview():
    """Create a visual overview of the dataset"""
    objects = ["book", "mug", "bottle", "toy", "remote"]
    rows = []

    for obj in objects:
        obj_dir = DATASET_DIR / obj
        ref = cv2.imread(str(obj_dir / "reference.png"))
        tests = sorted([f for f in obj_dir.glob("test_*.png")])[:4]

        row_imgs = [ref]
        for t in tests:
            row_imgs.append(cv2.imread(str(t)))
        while len(row_imgs) < 5:
            row_imgs.append(np.ones((300, 300, 3), dtype=np.uint8) * 200)

        # Resize to uniform size
        row_imgs = [cv2.resize(img, (180, 180)) for img in row_imgs]

        # Add labels
        labeled = []
        labels = ["REFERENCE"] + [f"TEST {i+1}" for i in range(4)]
        for img, label in zip(row_imgs, labels):
            canvas = np.ones((200, 180, 3), dtype=np.uint8) * 255
            canvas[20:200, :] = img
            color = (0, 120, 0) if label == "REFERENCE" else (120, 80, 0)
            cv2.putText(canvas, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            labeled.append(canvas)

        row = np.hstack(labeled)

        # Add object name label on left
        label_col = np.ones((200, 80, 3), dtype=np.uint8) * 245
        cv2.putText(label_col, obj.upper(), (5, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 100), 2)
        row = np.hstack([label_col, row])
        rows.append(row)

    # Add header
    header = np.ones((40, rows[0].shape[1], 3), dtype=np.uint8) * 40
    cv2.putText(header, "DATASET PREVIEW - Object Matching System",
                (20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 50), 2)

    preview = np.vstack([header] + rows)
    cv2.imwrite("results/dataset_preview.png", preview)
    print("\n[+] Dataset preview saved: results/dataset_preview.png")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING OBJECT MATCHING DATASET")
    print("=" * 60)
    os.makedirs("results", exist_ok=True)

    print("\nGenerating object images...")
    info = generate_dataset()

    print(f"\nDataset summary:")
    from collections import Counter
    obj_counts = Counter(i["object"] for i in info)
    for obj, cnt in obj_counts.items():
        print(f"  {obj}: {cnt} images (1 reference + {cnt-1} test)")

    print("\nCreating dataset preview...")
    create_dataset_preview()
    print("\n[✓] Dataset generation complete!")
