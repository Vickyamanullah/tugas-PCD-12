"""
Sistem Pencocokan Objek Berbasis Fitur Lokal
============================================
Complete implementation of:
1. Feature Detection & Description (SIFT, SURF, ORB)
2. Feature Matching (BF, FLANN, Lowe's ratio test, RANSAC)
3. Bag of Visual Words (BoVW) with SVM/k-NN
4. PCA Dimensionality Reduction
5. Comprehensive Evaluation
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import time
import os
import warnings
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from collections import defaultdict, Counter

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_recall_curve, average_precision_score)
from sklearn.model_selection import cross_val_score
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

OBJECTS = ["book", "mug", "bottle", "toy", "remote"]
COLORS = {
    "book": "#4E79A7", "mug": "#F28E2B", "bottle": "#59A14F",
    "toy": "#E15759", "remote": "#B07AA1"
}

# ============================================================
# SECTION 1: FEATURE DETECTION & DESCRIPTION
# ============================================================

def load_image_gray(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_image_color(path):
    return cv2.imread(str(path))


def create_detectors():
    """Initialize all feature detectors"""
    detectors = {}
    
    # SIFT
    try:
        detectors['SIFT'] = cv2.SIFT_create(nfeatures=500)
        print("  [✓] SIFT initialized")
    except Exception as e:
        print(f"  [✗] SIFT failed: {e}")

    # SURF
    try:
        detectors['SURF'] = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        print("  [✓] SURF initialized")
    except Exception as e:
        print(f"  [✗] SURF not available: {e}")

    # ORB
    try:
        detectors['ORB'] = cv2.ORB_create(nfeatures=500)
        print("  [✓] ORB initialized")
    except Exception as e:
        print(f"  [✗] ORB failed: {e}")

    return detectors


def detect_and_describe(detector, img_gray, name=""):
    """Detect keypoints and compute descriptors"""
    start = time.perf_counter()
    kps, descs = detector.detectAndCompute(img_gray, None)
    elapsed = (time.perf_counter() - start) * 1000  # ms

    if descs is None:
        descs = np.array([])

    return kps, descs, elapsed


def extract_all_features(detectors):
    """Extract features from all images for all detectors"""
    print("\n[1] FEATURE EXTRACTION")
    print("-" * 50)

    results = {name: {} for name in detectors}
    stats = {name: defaultdict(list) for name in detectors}

    for obj in OBJECTS:
        obj_dir = DATASET_DIR / obj
        all_imgs = sorted(obj_dir.glob("*.png"))

        for img_path in all_imgs:
            img = load_image_gray(img_path)
            img_key = f"{obj}/{img_path.name}"

            for det_name, detector in detectors.items():
                kps, descs, t = detect_and_describe(detector, img)
                results[det_name][img_key] = {
                    'kps': kps, 'descs': descs, 'time': t,
                    'n_kps': len(kps),
                    'desc_dim': descs.shape[1] if len(descs) > 0 else 0,
                    'path': str(img_path), 'object': obj,
                    'is_ref': img_path.name == 'reference.png'
                }
                stats[det_name]['n_kps'].append(len(kps))
                stats[det_name]['time'].append(t)

    # Print stats table
    print(f"\n{'Detector':<10} {'Avg KPs':>10} {'Min KPs':>10} {'Max KPs':>10} {'Avg Time(ms)':>14} {'Desc Dim':>10}")
    print("-" * 65)
    for det_name in detectors:
        kps_list = stats[det_name]['n_kps']
        t_list = stats[det_name]['time']
        # Get desc dim from first valid
        first_valid = next((v for v in results[det_name].values() if v['desc_dim'] > 0), None)
        dim = first_valid['desc_dim'] if first_valid else 0
        print(f"{det_name:<10} {np.mean(kps_list):>10.1f} {np.min(kps_list):>10} {np.max(kps_list):>10} {np.mean(t_list):>14.2f} {dim:>10}")

    return results, stats


def visualize_keypoints(detectors, results):
    """Create keypoint visualization for all detectors"""
    fig, axes = plt.subplots(len(detectors), len(OBJECTS), figsize=(18, 4 * len(detectors)))
    fig.suptitle("Feature Keypoints Visualization (Reference Images)", fontsize=14, fontweight='bold', y=1.01)

    for di, det_name in enumerate(detectors):
        for oi, obj in enumerate(OBJECTS):
            ax = axes[di][oi] if len(detectors) > 1 else axes[oi]
            img_key = f"{obj}/reference.png"
            data = results[det_name][img_key]
            img_color = load_image_color(data['path'])
            img_kp = cv2.drawKeypoints(img_color, data['kps'], None,
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            ax.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{det_name} - {obj}\n({data['n_kps']} KPs, {data['time']:.1f}ms)",
                         fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_keypoints_visualization.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 01_keypoints_visualization.png")


# ============================================================
# SECTION 2: FEATURE MATCHING
# ============================================================

def create_matchers():
    """Create BF and FLANN matchers"""
    matchers = {}

    # Brute-Force for L2 (SIFT/SURF)
    matchers['BF_L2'] = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # Brute-Force for Hamming (ORB)
    matchers['BF_HAM'] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # FLANN for SIFT/SURF (float descriptors)
    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matchers['FLANN_L2'] = cv2.FlannBasedMatcher(flann_params, search_params)

    # FLANN for ORB (binary descriptors)
    FLANN_INDEX_LSH = 6
    lsh_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    matchers['FLANN_HAM'] = cv2.FlannBasedMatcher(lsh_params, search_params)

    return matchers


def lowe_ratio_test(matches, ratio=0.75):
    """Apply Lowe's ratio test to filter matches"""
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def ransac_homography(kps1, kps2, good_matches, reproj_thresh=5.0):
    """Estimate homography using RANSAC"""
    if len(good_matches) < 4:
        return None, 0

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)
        if mask is not None:
            inliers = int(mask.sum())
            return H, inliers
    except:
        pass
    return None, 0


def run_matching_experiments(detectors, results):
    """Run all matching experiments"""
    print("\n[2] FEATURE MATCHING EXPERIMENTS")
    print("-" * 50)

    matchers = create_matchers()
    matching_results = []

    det_matcher_map = {
        'SIFT': ('BF_L2', 'FLANN_L2'),
        'SURF': ('BF_L2', 'FLANN_L2'),
        'ORB': ('BF_HAM', 'FLANN_HAM'),
    }

    for det_name, matcher_keys in det_matcher_map.items():
        if det_name not in detectors:
            continue

        for obj in OBJECTS:
            ref_key = f"{obj}/reference.png"
            ref_data = results[det_name][ref_key]

            for img_key, img_data in results[det_name].items():
                if img_data['is_ref'] or img_data['object'] != obj:
                    continue

                for m_key in matcher_keys:
                    matcher = matchers[m_key]

                    # Check valid descriptors
                    if len(ref_data['descs']) == 0 or len(img_data['descs']) == 0:
                        continue

                    try:
                        # kNN matching (k=2 for ratio test)
                        start = time.perf_counter()
                        raw_matches = matcher.knnMatch(ref_data['descs'], img_data['descs'], k=2)
                        t_match = (time.perf_counter() - start) * 1000

                        # Lowe's ratio test
                        good = lowe_ratio_test(raw_matches, ratio=0.75)

                        # RANSAC
                        _, inliers = ransac_homography(ref_data['kps'], img_data['kps'], good)

                        # Compute scores
                        total_kps = len(ref_data['kps'])
                        ratio_good = len(good) / max(len(raw_matches), 1)
                        inlier_ratio = inliers / max(len(good), 1)

                        # Determine variation type
                        var_type = "unknown"
                        for vt in ["rotation", "scale", "illumination", "occlusion", "combined"]:
                            if vt in img_key:
                                var_type = vt
                                break

                        matching_results.append({
                            'detector': det_name,
                            'matcher': m_key,
                            'object': obj,
                            'variation': var_type,
                            'raw_matches': len(raw_matches),
                            'good_matches': len(good),
                            'inliers': inliers,
                            'ratio_good': ratio_good,
                            'inlier_ratio': inlier_ratio,
                            'match_time': t_match,
                            'img_key': img_key
                        })
                    except Exception as e:
                        pass

    # Print summary
    from collections import defaultdict
    summary = defaultdict(lambda: defaultdict(list))
    for r in matching_results:
        key = (r['detector'], r['matcher'])
        summary[key]['good_matches'].append(r['good_matches'])
        summary[key]['inlier_ratio'].append(r['inlier_ratio'])
        summary[key]['match_time'].append(r['match_time'])

    print(f"\n{'Detector+Matcher':<20} {'Avg Good':>10} {'Avg Inlier%':>12} {'Avg Time(ms)':>13}")
    print("-" * 58)
    for (det, mat), vals in sorted(summary.items()):
        print(f"{det+'+'+mat:<20} {np.mean(vals['good_matches']):>10.1f} "
              f"{np.mean(vals['inlier_ratio'])*100:>11.1f}% "
              f"{np.mean(vals['match_time']):>13.2f}")

    return matching_results


def visualize_matches(detectors, results, n_examples=3):
    """Visualize best matches for each detector"""
    det_matcher_map = {
        'SIFT': ('BF_L2', cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)),
        'SURF': ('BF_L2', cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)),
        'ORB': ('BF_HAM', cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)),
    }

    available = [(d, m) for d, m in det_matcher_map.items() if d in detectors]
    n_det = len(available)

    fig, axes = plt.subplots(n_det, n_examples, figsize=(16, 5 * n_det))
    if n_det == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Feature Matching Results (Lowe's Ratio Test + RANSAC)", fontsize=13, fontweight='bold')

    objs_to_show = OBJECTS[:n_examples]

    for di, (det_name, (m_name, matcher)) in enumerate(available):
        for oi, obj in enumerate(objs_to_show):
            ax = axes[di][oi]
            ref_key = f"{obj}/reference.png"
            test_key = f"{obj}/test_01_rotation.png"

            ref_data = results[det_name][ref_key]
            test_data = results[det_name].get(test_key)

            if test_data is None or len(ref_data['descs']) == 0 or len(test_data['descs']) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue

            try:
                raw = matcher.knnMatch(ref_data['descs'], test_data['descs'], k=2)
                good = lowe_ratio_test(raw)

                ref_img = load_image_color(ref_data['path'])
                test_img = load_image_color(test_data['path'])

                draw_params = dict(
                    matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                match_img = cv2.drawMatches(ref_img, ref_data['kps'],
                                             test_img, test_data['kps'],
                                             good[:30], None, **draw_params)
                ax.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
                _, inliers = ransac_homography(ref_data['kps'], test_data['kps'], good)
                ax.set_title(f"{det_name} | {obj} | {len(good)} matches | {inliers} inliers", fontsize=9)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center', fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_matching_visualization.png", dpi=110, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 02_matching_visualization.png")


def plot_precision_recall(matching_results):
    """Plot precision-recall curves per detector"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Precision-Recall Analysis by Detector & Variation", fontsize=13, fontweight='bold')

    detectors_pr = ['SIFT', 'SURF', 'ORB']
    variations = ['rotation', 'scale', 'illumination', 'occlusion', 'combined']
    var_colors = plt.cm.Set1(np.linspace(0, 0.9, len(variations)))

    for ax, det in zip(axes, detectors_pr):
        det_results = [r for r in matching_results if r['detector'] == det]

        for vi, var in enumerate(variations):
            var_results = [r for r in det_results if r['variation'] == var]
            if not var_results:
                continue

            # Use inlier_ratio as precision proxy, ratio_good as recall proxy
            precisions = [r['inlier_ratio'] for r in var_results]
            recalls = [r['ratio_good'] for r in var_results]

            if len(precisions) > 1:
                sorted_idx = np.argsort(recalls)
                ax.plot(np.array(recalls)[sorted_idx], np.array(precisions)[sorted_idx],
                        'o-', color=var_colors[vi], label=var, linewidth=2, markersize=5, alpha=0.8)
            elif len(precisions) == 1:
                ax.scatter(recalls, precisions, color=var_colors[vi], label=var, s=80)

        ax.set_xlabel("Recall (Good Match Ratio)", fontsize=10)
        ax.set_ylabel("Precision (Inlier Ratio)", fontsize=10)
        ax.set_title(f"{det} Detector", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_precision_recall_curves.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 03_precision_recall_curves.png")


# ============================================================
# SECTION 3: BAG OF VISUAL WORDS
# ============================================================

def build_bovw_system(detectors, vocab_sizes=[10, 20, 50, 100]):
    """Build and evaluate BoVW system"""
    print("\n[3] BAG OF VISUAL WORDS (BoVW)")
    print("-" * 50)

    # Collect all descriptors from training (reference) images
    all_descriptors = {det: [] for det in detectors}
    train_data = {det: {'descs': [], 'labels': []} for det in detectors}

    for obj_idx, obj in enumerate(OBJECTS):
        obj_dir = DATASET_DIR / obj
        all_imgs = sorted(obj_dir.glob("*.png"))

        for img_path in all_imgs:
            img = load_image_gray(img_path)
            label = obj_idx

            for det_name, detector in detectors.items():
                _, descs, _ = detect_and_describe(detector, img)
                if len(descs) > 0:
                    all_descriptors[det_name].append(descs)
                    train_data[det_name]['descs'].append(descs)
                    train_data[det_name]['labels'].append(label)

    bovw_results = {}

    for det_name in detectors:
        if not all_descriptors[det_name]:
            continue

        print(f"\n  Processing {det_name}...")
        desc_stack = np.vstack(all_descriptors[det_name]).astype(np.float32)
        bovw_results[det_name] = {}

        for k in vocab_sizes:
            print(f"    Vocabulary size k={k}...", end=' ', flush=True)
            start = time.perf_counter()

            # K-Means clustering
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
            kmeans.fit(desc_stack)
            t_kmeans = time.perf_counter() - start

            # Build histograms for all images
            histograms = []
            labels = []

            for img_descs, label in zip(train_data[det_name]['descs'],
                                         train_data[det_name]['labels']):
                if len(img_descs) > 0:
                    words = kmeans.predict(img_descs.astype(np.float32))
                    hist, _ = np.histogram(words, bins=k, range=(0, k))
                    hist = hist.astype(np.float32)
                    hist /= (hist.sum() + 1e-7)  # Normalize
                    histograms.append(hist)
                    labels.append(label)
                else:
                    histograms.append(np.zeros(k))
                    labels.append(label)

            X = np.array(histograms)
            y = np.array(labels)

            # Train classifiers
            # SVM
            svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
            svm_scores = cross_val_score(svm, X, y, cv=min(5, len(set(y))), scoring='accuracy')

            # k-NN
            knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
            knn_scores = cross_val_score(knn, X, y, cv=min(5, len(set(y))), scoring='accuracy')

            bovw_results[det_name][k] = {
                'kmeans': kmeans,
                'svm_acc': svm_scores.mean(),
                'svm_std': svm_scores.std(),
                'knn_acc': knn_scores.mean(),
                'knn_std': knn_scores.std(),
                'vocab_size': k,
                't_kmeans': t_kmeans,
                'X': X, 'y': y
            }

            print(f"SVM={svm_scores.mean():.3f}±{svm_scores.std():.3f}  "
                  f"kNN={knn_scores.mean():.3f}±{knn_scores.std():.3f}  "
                  f"({t_kmeans:.1f}s)")

    return bovw_results


def visualize_bovw(bovw_results):
    """Visualize BoVW results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bag of Visual Words - Classification Performance", fontsize=13, fontweight='bold')

    det_colors = {'SIFT': '#4E79A7', 'SURF': '#F28E2B', 'ORB': '#59A14F'}
    vocab_sizes = [10, 20, 50, 100]

    # SVM accuracy vs vocab size
    ax = axes[0][0]
    for det_name, k_results in bovw_results.items():
        accs = [k_results[k]['svm_acc'] for k in vocab_sizes if k in k_results]
        ks = [k for k in vocab_sizes if k in k_results]
        if accs:
            ax.plot(ks, accs, 'o-', label=det_name, color=det_colors.get(det_name, 'gray'),
                    linewidth=2.5, markersize=8)
    ax.set_xlabel("Vocabulary Size (k)", fontsize=11)
    ax.set_ylabel("SVM Accuracy", fontsize=11)
    ax.set_title("SVM Accuracy vs Vocabulary Size", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # kNN accuracy vs vocab size
    ax = axes[0][1]
    for det_name, k_results in bovw_results.items():
        accs = [k_results[k]['knn_acc'] for k in vocab_sizes if k in k_results]
        ks = [k for k in vocab_sizes if k in k_results]
        if accs:
            ax.plot(ks, accs, 's--', label=det_name, color=det_colors.get(det_name, 'gray'),
                    linewidth=2.5, markersize=8)
    ax.set_xlabel("Vocabulary Size (k)", fontsize=11)
    ax.set_ylabel("k-NN Accuracy", fontsize=11)
    ax.set_title("k-NN Accuracy vs Vocabulary Size", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Confusion matrix for best configuration
    best_det = None
    best_acc = 0
    best_k = 50

    for det_name, k_results in bovw_results.items():
        if best_k in k_results:
            acc = k_results[best_k]['svm_acc']
            if acc > best_acc:
                best_acc = acc
                best_det = det_name

    if best_det and best_k in bovw_results[best_det]:
        ax = axes[1][0]
        data = bovw_results[best_det][best_k]
        X, y = data['X'], data['y']

        # Train final SVM and predict
        svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        svm.fit(X, y)
        y_pred = svm.predict(X)
        cm = confusion_matrix(y, y_pred)

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"Confusion Matrix (SVM, {best_det}, k={best_k})", fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.85)
        tick_marks = np.arange(len(OBJECTS))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([o[:4] for o in OBJECTS], rotation=45, fontsize=9)
        ax.set_yticklabels([o[:4] for o in OBJECTS], fontsize=9)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)

        # Annotate cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=11)

    # Comparison bar chart
    ax = axes[1][1]
    det_names = list(bovw_results.keys())
    x = np.arange(len(det_names))
    width = 0.35

    svm_accs = [bovw_results[d].get(50, {}).get('svm_acc', 0) for d in det_names]
    knn_accs = [bovw_results[d].get(50, {}).get('knn_acc', 0) for d in det_names]

    bars1 = ax.bar(x - width/2, svm_accs, width, label='SVM', color='#4E79A7', alpha=0.85)
    bars2 = ax.bar(x + width/2, knn_accs, width, label='k-NN', color='#F28E2B', alpha=0.85)

    ax.set_xlabel("Detector", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("SVM vs k-NN Accuracy (k=50)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(det_names)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_bovw_results.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 04_bovw_results.png")


# ============================================================
# SECTION 4: PCA DIMENSIONALITY REDUCTION
# ============================================================

def pca_analysis(detectors, results):
    """Apply PCA to descriptors and analyze impact on matching"""
    print("\n[4] PCA DIMENSIONALITY REDUCTION")
    print("-" * 50)

    components_list = [16, 32, 64, 128]
    pca_results = {}

    for det_name, detector in detectors.items():
        print(f"\n  {det_name}:")
        
        # Collect all descriptors
        all_descs = []
        ref_descs_by_obj = {}
        test_descs_by_obj = defaultdict(list)
        ref_kps_by_obj = {}
        test_kps_by_obj = defaultdict(list)

        for obj in OBJECTS:
            for img_key, data in results[det_name].items():
                if data['object'] != obj or len(data['descs']) == 0:
                    continue
                all_descs.append(data['descs'])
                if data['is_ref']:
                    ref_descs_by_obj[obj] = data['descs']
                    ref_kps_by_obj[obj] = data['kps']
                else:
                    test_descs_by_obj[obj].append(data['descs'])
                    test_kps_by_obj[obj].append(data['kps'])

        if not all_descs:
            continue

        desc_stack = np.vstack(all_descs).astype(np.float64)
        orig_dim = desc_stack.shape[1]

        # Standardize
        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc_stack)

        pca_results[det_name] = {}
        matcher_l2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Baseline (no PCA)
        base_matches = _compute_matching_score(
            ref_descs_by_obj, test_descs_by_obj,
            ref_kps_by_obj, test_kps_by_obj, matcher_l2
        )

        pca_results[det_name]['original'] = {
            'n_components': orig_dim,
            'explained_var': 1.0,
            'matching_score': base_matches,
            'compression': 1.0
        }

        print(f"    Original ({orig_dim}D): matching_score={base_matches:.3f}")

        # Fit PCA on full data
        max_comp = min(512, desc_scaled.shape[0], desc_scaled.shape[1])
        pca_full = PCA(n_components=max_comp, random_state=42)
        pca_full.fit(desc_scaled[:min(5000, desc_scaled.shape[0])])

        for n_comp in components_list:
            safe_n = min(n_comp, orig_dim - 1, desc_scaled.shape[0] - 1)
            if safe_n < 2:
                continue
            n_comp = safe_n

            # Fit PCA globally on all descriptors
            all_scaled = scaler.transform(desc_stack)
            pca_refit = PCA(n_components=n_comp, random_state=42)
            pca_refit.fit(all_scaled[:min(5000, all_scaled.shape[0])])

            ref_pca_final = {}
            test_pca_final = defaultdict(list)
            for obj in OBJECTS:
                if obj in ref_descs_by_obj:
                    sc = scaler.transform(ref_descs_by_obj[obj].astype(np.float64))
                    ref_pca_final[obj] = pca_refit.transform(sc).astype(np.float32)
                for td in test_descs_by_obj[obj]:
                    sc = scaler.transform(td.astype(np.float64))
                    test_pca_final[obj].append(pca_refit.transform(sc).astype(np.float32))

            # Compute matching score with PCA descriptors
            pca_match_score = _compute_matching_score(
                ref_pca_final, test_pca_final,
                ref_kps_by_obj, test_kps_by_obj, matcher_l2
            )

            explained = pca_refit.explained_variance_ratio_.sum()
            compression = n_comp / orig_dim

            pca_results[det_name][n_comp] = {
                'n_components': n_comp,
                'explained_var': explained,
                'matching_score': pca_match_score,
                'compression': compression
            }

            print(f"    PCA-{n_comp:3d}: explained={explained:.3f}  "
                  f"matching={pca_match_score:.3f}  compression={compression:.3f}")

    return pca_results


def _compute_matching_score(ref_descs, test_descs, ref_kps, test_kps, matcher):
    """Compute average inlier ratio across all object-test pairs"""
    scores = []
    for obj in OBJECTS:
        if obj not in ref_descs or not test_descs[obj]:
            continue
        rd = ref_descs[obj].astype(np.float32)
        rk = ref_kps[obj]
        for td, tk in zip(test_descs[obj], test_kps[obj]):
            if len(td) == 0:
                continue
            td = td.astype(np.float32)
            try:
                raw = matcher.knnMatch(rd, td, k=2)
                good = lowe_ratio_test(raw)
                _, inliers = ransac_homography(rk, tk, good)
                score = inliers / max(len(good), 1)
                scores.append(score)
            except:
                pass
    return np.mean(scores) if scores else 0.0


def visualize_pca(pca_results):
    """Visualize PCA analysis"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PCA Dimensionality Reduction Analysis", fontsize=13, fontweight='bold')

    det_colors = {'SIFT': '#4E79A7', 'SURF': '#F28E2B', 'ORB': '#59A14F'}
    components_list = [16, 32, 64, 128]

    # Plot 1: Matching score vs n_components
    ax = axes[0]
    for det_name, comp_results in pca_results.items():
        comp_keys = [k for k in comp_results if k != 'original' and isinstance(k, int)]
        if not comp_keys:
            continue
        comp_keys.sort()
        scores = [comp_results[k]['matching_score'] for k in comp_keys]
        orig_score = comp_results['original']['matching_score']

        ax.axhline(orig_score, linestyle='--', color=det_colors.get(det_name, 'gray'),
                   alpha=0.5, label=f'{det_name} (orig)')
        ax.plot(comp_keys, scores, 'o-', color=det_colors.get(det_name, 'gray'),
                label=f'{det_name} (PCA)', linewidth=2.5, markersize=8)

    ax.set_xlabel("PCA Components", fontsize=11)
    ax.set_ylabel("Matching Score (Inlier Ratio)", fontsize=11)
    ax.set_title("Matching Accuracy vs PCA Components", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Explained variance
    ax = axes[1]
    for det_name, comp_results in pca_results.items():
        comp_keys = [k for k in comp_results if k != 'original' and isinstance(k, int)]
        if not comp_keys:
            continue
        comp_keys.sort()
        ev = [comp_results[k]['explained_var'] for k in comp_keys]
        ax.plot(comp_keys, ev, 's-', color=det_colors.get(det_name, 'gray'),
                label=det_name, linewidth=2.5, markersize=8)

    ax.set_xlabel("PCA Components", fontsize=11)
    ax.set_ylabel("Explained Variance Ratio", fontsize=11)
    ax.set_title("Explained Variance vs Components", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.95, linestyle=':', color='red', alpha=0.7, label='95%')

    # Plot 3: Accuracy vs Compression tradeoff
    ax = axes[2]
    for det_name, comp_results in pca_results.items():
        comp_keys = [k for k in comp_results if k != 'original' and isinstance(k, int)]
        if not comp_keys:
            continue
        comp_keys.sort()

        compressions = [comp_results[k]['compression'] for k in comp_keys]
        scores = [comp_results[k]['matching_score'] for k in comp_keys]
        orig = comp_results['original']

        ax.scatter([orig['compression']], [orig['matching_score']],
                   color=det_colors.get(det_name, 'gray'), s=150, marker='*', zorder=5)
        for ci, (c, s, k) in enumerate(zip(compressions, scores, comp_keys)):
            ax.scatter([c], [s], color=det_colors.get(det_name, 'gray'), s=80, alpha=0.8)
            ax.annotate(f'{k}D', (c, s), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        ax.plot(compressions, scores, '-', color=det_colors.get(det_name, 'gray'),
                alpha=0.6, linewidth=1.5, label=det_name)

    ax.set_xlabel("Compression Ratio (PCA/Original)", fontsize=11)
    ax.set_ylabel("Matching Score", fontsize=11)
    ax.set_title("Accuracy vs Compression Tradeoff", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "05_pca_analysis.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 05_pca_analysis.png")


# ============================================================
# SECTION 5: COMPREHENSIVE EVALUATION
# ============================================================

def comprehensive_comparison(results, matching_results, bovw_results, pca_results, stats):
    """Generate comprehensive comparison table and charts"""
    print("\n[5] COMPREHENSIVE EVALUATION")
    print("-" * 50)

    # Speed comparison
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Comprehensive Method Comparison", fontsize=15, fontweight='bold')

    det_colors = {'SIFT': '#4E79A7', 'SURF': '#F28E2B', 'ORB': '#59A14F'}
    variations = ['rotation', 'scale', 'illumination', 'occlusion', 'combined']

    # 1. Extraction speed comparison
    ax1 = fig.add_subplot(gs[0, 0])
    det_names = list(stats.keys())
    avg_times = [np.mean(stats[d]['time']) for d in det_names]
    std_times = [np.std(stats[d]['time']) for d in det_names]
    colors_bar = [det_colors.get(d, 'gray') for d in det_names]
    bars = ax1.bar(det_names, avg_times, color=colors_bar, alpha=0.85,
                   yerr=std_times, capsize=5)
    ax1.set_title("Feature Extraction Speed", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Time (ms)", fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, avg_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}ms', ha='center', va='bottom', fontsize=9)

    # 2. Keypoint count comparison
    ax2 = fig.add_subplot(gs[0, 1])
    avg_kps = [np.mean(stats[d]['n_kps']) for d in det_names]
    ax2.bar(det_names, avg_kps, color=colors_bar, alpha=0.85)
    ax2.set_title("Average Keypoint Count", fontsize=11, fontweight='bold')
    ax2.set_ylabel("# Keypoints", fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (d, val) in enumerate(zip(det_names, avg_kps)):
        ax2.text(i, val + 2, f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    # 3. Robustness by variation type
    ax3 = fig.add_subplot(gs[0, 2])
    var_scores = {d: {} for d in det_names}
    for r in matching_results:
        d = r['detector']
        v = r['variation']
        if d not in var_scores:
            continue
        if v not in var_scores[d]:
            var_scores[d][v] = []
        var_scores[d][v].append(r['inlier_ratio'])

    x = np.arange(len(variations))
    width = 0.25
    for i, det in enumerate(det_names):
        scores = [np.mean(var_scores[det].get(v, [0])) for v in variations]
        offset = (i - len(det_names)/2 + 0.5) * width
        ax3.bar(x + offset, scores, width, label=det,
                color=det_colors.get(det, 'gray'), alpha=0.85)

    ax3.set_xticks(x)
    ax3.set_xticklabels([v[:4] for v in variations], fontsize=9)
    ax3.set_title("Robustness by Variation Type", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Avg Inlier Ratio", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Good matches by object
    ax4 = fig.add_subplot(gs[1, 0])
    obj_match_scores = {d: {} for d in det_names}
    for r in matching_results:
        d = r['detector']
        o = r['object']
        if d not in obj_match_scores:
            continue
        if o not in obj_match_scores[d]:
            obj_match_scores[d][o] = []
        obj_match_scores[d][o].append(r['good_matches'])

    x = np.arange(len(OBJECTS))
    for i, det in enumerate(det_names):
        scores = [np.mean(obj_match_scores[det].get(o, [0])) for o in OBJECTS]
        offset = (i - len(det_names)/2 + 0.5) * width
        ax4.bar(x + offset, scores, width, label=det,
                color=det_colors.get(det, 'gray'), alpha=0.85)

    ax4.set_xticks(x)
    ax4.set_xticklabels(OBJECTS, fontsize=9)
    ax4.set_title("Good Matches by Object Class", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Avg Good Matches", fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. BoVW accuracy summary
    ax5 = fig.add_subplot(gs[1, 1])
    vocab_sizes_plot = [10, 20, 50, 100]
    for det_name, k_results in bovw_results.items():
        svm_accs = [k_results.get(k, {}).get('svm_acc', 0) for k in vocab_sizes_plot]
        knn_accs = [k_results.get(k, {}).get('knn_acc', 0) for k in vocab_sizes_plot]
        ks = vocab_sizes_plot
        ax5.plot(ks, svm_accs, 'o-', color=det_colors.get(det_name, 'gray'),
                 label=f'{det_name} SVM', linewidth=2)
        ax5.plot(ks, knn_accs, 's--', color=det_colors.get(det_name, 'gray'),
                 label=f'{det_name} kNN', linewidth=1.5, alpha=0.7)
    ax5.set_title("BoVW Classification Accuracy", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Vocabulary Size", fontsize=10)
    ax5.set_ylabel("Accuracy", fontsize=10)
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.05)

    # 6. Summary radar-like comparison
    ax6 = fig.add_subplot(gs[1, 2])
    categories = ['Speed\n(inv)', 'KP Count\n(norm)', 'BF Match\nAccuracy',
                  'FLANN Match\nAccuracy', 'BoVW\nAccuracy']

    # Normalize metrics for comparison
    summary_scores = {}
    for det in det_names:
        speed_inv = 1 / max(np.mean(stats[det]['time']), 0.01)
        kp_norm = np.mean(stats[det]['n_kps']) / 500

        bf_key = 'BF_L2' if det != 'ORB' else 'BF_HAM'
        flann_key = 'FLANN_L2' if det != 'ORB' else 'FLANN_HAM'

        bf_acc = np.mean([r['inlier_ratio'] for r in matching_results
                          if r['detector'] == det and r['matcher'] == bf_key] or [0])
        flann_acc = np.mean([r['inlier_ratio'] for r in matching_results
                             if r['detector'] == det and r['matcher'] == flann_key] or [0])

        bovw_acc = bovw_results.get(det, {}).get(50, {}).get('svm_acc', 0)

        summary_scores[det] = [speed_inv, kp_norm, bf_acc, flann_acc, bovw_acc]

    # Normalize all to 0-1
    all_vals = np.array(list(summary_scores.values()))
    max_vals = all_vals.max(axis=0)
    for det in summary_scores:
        summary_scores[det] = [v / max(m, 1e-6) for v, m in
                                zip(summary_scores[det], max_vals)]

    x = np.arange(len(categories))
    for i, det in enumerate(det_names):
        offset = (i - len(det_names)/2 + 0.5) * 0.25
        ax6.bar(x + offset, summary_scores[det], 0.25, label=det,
                color=det_colors.get(det, 'gray'), alpha=0.85)

    ax6.set_xticks(x)
    ax6.set_xticklabels(categories, fontsize=8)
    ax6.set_title("Normalized Performance Summary", fontsize=11, fontweight='bold')
    ax6.set_ylabel("Normalized Score", fontsize=10)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim(0, 1.2)

    # 7. PCA compression-accuracy tradeoff (heatmap style)
    ax7 = fig.add_subplot(gs[2, :])

    # Create comparison table as text
    table_data = []
    headers = ['Detector', 'Avg KPs', 'Ext Time (ms)', 'Desc Dim',
               'BF Match Score', 'FLANN Match Score', 'BoVW SVM (k=50)', 'BoVW kNN (k=50)',
               'PCA-64 Acc', 'Best Variation']

    for det in det_names:
        avg_kps = f"{np.mean(stats[det]['n_kps']):.0f}"
        ext_t = f"{np.mean(stats[det]['time']):.2f}"
        
        # Desc dim
        first_valid = next((v for v in results[det].values() if v['desc_dim'] > 0), None)
        desc_dim = str(first_valid['desc_dim']) if first_valid else '-'

        bf_key = 'BF_L2' if det != 'ORB' else 'BF_HAM'
        flann_key = 'FLANN_L2' if det != 'ORB' else 'FLANN_HAM'

        bf_score = np.mean([r['inlier_ratio'] for r in matching_results
                            if r['detector'] == det and r['matcher'] == bf_key] or [0])
        flann_score = np.mean([r['inlier_ratio'] for r in matching_results
                               if r['detector'] == det and r['matcher'] == flann_key] or [0])

        bovw_svm = bovw_results.get(det, {}).get(50, {}).get('svm_acc', 0)
        bovw_knn = bovw_results.get(det, {}).get(50, {}).get('knn_acc', 0)

        pca64 = pca_results.get(det, {}).get(64, {}).get('matching_score', 0)

        # Best variation
        var_means = {}
        for v in variations:
            v_scores = [r['inlier_ratio'] for r in matching_results
                        if r['detector'] == det and r['variation'] == v]
            if v_scores:
                var_means[v] = np.mean(v_scores)
        best_var = max(var_means, key=var_means.get) if var_means else '-'

        table_data.append([det, avg_kps, ext_t, desc_dim,
                           f'{bf_score:.3f}', f'{flann_score:.3f}',
                           f'{bovw_svm:.3f}', f'{bovw_knn:.3f}',
                           f'{pca64:.3f}', best_var])

    ax7.axis('tight')
    ax7.axis('off')
    table = ax7.table(cellText=table_data, colLabels=headers,
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color rows alternately
    row_colors = ['#EAF2FB', '#FDFEFE']
    for i, row in enumerate(table_data):
        for j in range(len(headers)):
            table[i+1, j].set_facecolor(row_colors[i % 2])

    ax7.set_title("Comprehensive Comparison Table", fontsize=12, fontweight='bold', pad=20)

    plt.savefig(RESULTS_DIR / "06_comprehensive_comparison.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("[+] Saved: 06_comprehensive_comparison.png")

    # Print table to console
    print("\n  COMPARISON TABLE:")
    print(f"  {'Detector':<8} {'Avg KPs':>8} {'Time(ms)':>10} {'Dim':>6} "
          f"{'BF Score':>10} {'FLANN':>8} {'BoVW-SVM':>10} {'PCA-64':>8}")
    print("  " + "-" * 72)
    for row in table_data:
        print(f"  {row[0]:<8} {row[1]:>8} {row[2]:>10} {row[3]:>6} "
              f"{row[4]:>10} {row[5]:>8} {row[6]:>10} {row[8]:>8}")


def generate_analysis_report(matching_results, bovw_results, pca_results, stats):
    """Generate text analysis report"""
    report = []
    report.append("=" * 70)
    report.append("LAPORAN ANALISIS: SISTEM PENCOCOKAN OBJEK BERBASIS FITUR LOKAL")
    report.append("=" * 70)

    det_names = list(stats.keys())

    report.append("\n1. ANALISIS KECEPATAN DAN EFISIENSI")
    report.append("-" * 40)
    fastest = min(det_names, key=lambda d: np.mean(stats[d]['time']))
    slowest = max(det_names, key=lambda d: np.mean(stats[d]['time']))
    report.append(f"  Tercepat: {fastest} ({np.mean(stats[fastest]['time']):.2f}ms avg)")
    report.append(f"  Terlambat: {slowest} ({np.mean(stats[slowest]['time']):.2f}ms avg)")
    report.append(f"  Rasio kecepatan {fastest}/{slowest}: "
                  f"{np.mean(stats[slowest]['time'])/np.mean(stats[fastest]['time']):.1f}x lebih cepat")

    report.append("\n2. AKURASI PENCOCOKAN FITUR")
    report.append("-" * 40)
    for det in det_names:
        bf_key = 'BF_L2' if det != 'ORB' else 'BF_HAM'
        scores = [r['inlier_ratio'] for r in matching_results
                  if r['detector'] == det and r['matcher'] == bf_key]
        if scores:
            report.append(f"  {det} (BF): avg inlier ratio = {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    report.append("\n3. ROBUSTNESS TERHADAP DEGRADASI")
    report.append("-" * 40)
    variations = ['rotation', 'scale', 'illumination', 'occlusion', 'combined']
    for v in variations:
        v_scores = {d: np.mean([r['inlier_ratio'] for r in matching_results
                                 if r['detector'] == d and r['variation'] == v] or [0])
                    for d in det_names}
        best = max(v_scores, key=v_scores.get)
        report.append(f"  {v:15s}: Best = {best} ({v_scores[best]:.3f})")

    report.append("\n4. BAG OF VISUAL WORDS")
    report.append("-" * 40)
    for det in det_names:
        if det not in bovw_results:
            continue
        best_k = max(bovw_results[det].keys(), key=lambda k: bovw_results[det][k]['svm_acc'])
        acc = bovw_results[det][best_k]['svm_acc']
        report.append(f"  {det}: Best SVM accuracy = {acc:.3f} (k={best_k})")

    report.append("\n5. PCA DIMENSIONALITY REDUCTION")
    report.append("-" * 40)
    for det in det_names:
        if det not in pca_results:
            continue
        orig = pca_results[det].get('original', {}).get('matching_score', 0)
        pca64 = pca_results[det].get(64, {}).get('matching_score', 0)
        if orig > 0:
            drop = (orig - pca64) / orig * 100
            report.append(f"  {det}: Original={orig:.3f}, PCA-64={pca64:.3f}, "
                          f"Drop={drop:.1f}%")

    report.append("\n6. REKOMENDASI PENGGUNAAN")
    report.append("-" * 40)
    report.append("  • SIFT: Direkomendasikan untuk akurasi tinggi, tolerate terhadap")
    report.append("          variasi skala/rotasi, cocok untuk pencocokan referensi")
    report.append("  • SURF: Keseimbangan kecepatan-akurasi yang baik, cocok untuk")
    report.append("          aplikasi real-time dengan resource memadai")
    report.append("  • ORB:  Tercepat dan gratis lisensi, cocok untuk embedded systems,")
    report.append("          perangkat mobile, dan aplikasi real-time")
    report.append("  • BoVW dengan k=50: Optimal untuk klasifikasi objek (SVM)")
    report.append("  • PCA-64: Kompromi baik antara akurasi dan efisiensi memori")

    report.append("\n" + "=" * 70)
    report_text = "\n".join(report)
    print(report_text)

    with open(RESULTS_DIR / "analysis_report.txt", 'w') as f:
        f.write(report_text)
    print("\n[+] Saved: analysis_report.txt")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  SISTEM PENCOCOKAN OBJEK BERBASIS FITUR LOKAL")
    print("=" * 70)

    # Initialize detectors
    print("\nInitializing feature detectors...")
    detectors = create_detectors()

    if not detectors:
        print("[!] No detectors available!")
        return

    # 1. Feature extraction
    results, stats = extract_all_features(detectors)
    visualize_keypoints(detectors, results)

    # 2. Feature matching
    matching_results = run_matching_experiments(detectors, results)
    visualize_matches(detectors, results)
    plot_precision_recall(matching_results)

    # 3. Bag of Visual Words
    bovw_results = build_bovw_system(detectors, vocab_sizes=[10, 20, 50, 100])
    visualize_bovw(bovw_results)

    # 4. PCA analysis
    pca_results = pca_analysis(detectors, results)
    visualize_pca(pca_results)

    # 5. Comprehensive evaluation
    comprehensive_comparison(results, matching_results, bovw_results, pca_results, stats)

    # Generate report
    generate_analysis_report(matching_results, bovw_results, pca_results, stats)

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print(f"  All results saved to: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
