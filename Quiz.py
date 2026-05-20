import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def latihan_1():
    print("COMPARISON OF LOCAL FEATURE DESCRIPTORS")
    print("=" * 50)
    
    # Buat citra test dengan transformasi
    def create_test_images():
        # Citra dasar dengan pattern yang jelas
        img1 = np.zeros((300, 400), dtype=np.uint8)
        
        # Tambahkan shapes dengan corner features
        cv2.rectangle(img1, (50, 50), (150, 150), 200, -1)  # Square
        cv2.circle(img1, (250, 100), 50, 150, -1)           # Circle
        cv2.rectangle(img1, (100, 200), (200, 250), 100, -1) # Rectangle
        cv2.line(img1, (300, 50), (350, 200), 180, 3)       # Diagonal line
        
        # Citra kedua dengan transformasi
        # 1. Rotation
        M_rotate = cv2.getRotationMatrix2D((200, 150), 30, 1.0)
        img2 = cv2.warpAffine(img1, M_rotate, (400, 300))
        
        # 2. Scale
        img2 = cv2.resize(img2, None, fx=0.9, fy=0.9)
        
        # 3. Add noise
        noise = np.random.normal(0, 10, img2.shape)
        img2 = np.clip(img2.astype(float) + noise, 0, 255).astype(np.uint8)
        
        # 4. Brightness change
        img2 = cv2.convertScaleAbs(img2, alpha=1.2, beta=10)
        
        return img1, img2
    
    img1, img2 = create_test_images()
    
    # Inisialisasi feature detectors
    try:
        # SIFT
        sift = cv2.SIFT_create()
        
        # SURF (jika tersedia)
        try:
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
            surf_status = "Available"
        except:
            surf = None
            surf_status = "Not Available"
        
        # ORB
        orb = cv2.ORB_create(nfeatures=500)
        
    except Exception as e:
        print(f"Error initializing feature detectors: {e}")
        return
    
    # Function untuk ekstraksi dan visualisasi features
    def extract_and_display_features(detector, img, detector_name):
        if detector is None:
            return None, None, None
        
        # Detect keypoints dan compute descriptors
        keypoints, descriptors = detector.detectAndCompute(img, None)
        
        # Draw keypoints
        img_with_kp = cv2.drawKeypoints(img, keypoints, None, 
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return keypoints, descriptors, img_with_kp
    
    # Ekstraksi features dari kedua citra
    print("\nFEATURE EXTRACTION RESULTS:")
    print("-" * 40)
    
    results = {}
    
    # SIFT
    kp1_sift, desc1_sift, img1_sift = extract_and_display_features(sift, img1, "SIFT")
    kp2_sift, desc2_sift, img2_sift = extract_and_display_features(sift, img2, "SIFT")
    if kp1_sift is not None:
        results['SIFT'] = {
            'img1_kp': len(kp1_sift),
            'img2_kp': len(kp2_sift),
            'desc_dim': desc1_sift.shape[1] if desc1_sift is not None else 0,
            'desc_type': 'Float32'
        }
    
    # SURF
    if surf is not None:
        kp1_surf, desc1_surf, img1_surf = extract_and_display_features(surf, img1, "SURF")
        kp2_surf, desc2_surf, img2_surf = extract_and_display_features(surf, img2, "SURF")
        if kp1_surf is not None:
            results['SURF'] = {
                'img1_kp': len(kp1_surf),
                'img2_kp': len(kp2_surf),
                'desc_dim': desc1_surf.shape[1] if desc1_surf is not None else 0,
                'desc_type': 'Float32'
            }
    
    # ORB
    kp1_orb, desc1_orb, img1_orb = extract_and_display_features(orb, img1, "ORB")
    kp2_orb, desc2_orb, img2_orb = extract_and_display_features(orb, img2, "ORB")
    if kp1_orb is not None:
        results['ORB'] = {
            'img1_kp': len(kp1_orb),
            'img2_kp': len(kp2_orb),
            'desc_dim': desc1_orb.shape[1] if desc1_orb is not None else 0,
            'desc_type': 'Binary'
        }
    
    # Visualisasi results
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: SIFT
    if 'SIFT' in results:
        axes[0, 0].imshow(img1, cmap='gray')
        axes[0, 0].set_title('Image 1 (Original)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img1_sift, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'SIFT Keypoints\n{results["SIFT"]["img1_kp"]} points')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(img2_sift, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f'SIFT on Transformed\n{results["SIFT"]["img2_kp"]} points')
        axes[0, 2].axis('off')
    else:
        for i in range(3):
            axes[0, i].text(0.5, 0.5, 'SIFT Not Available', 
                          ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].axis('off')
    
    # Row 2: SURF
    if 'SURF' in results:
        axes[1, 0].imshow(img1, cmap='gray')
        axes[1, 0].set_title('Image 1 (Original)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(img1_surf, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'SURF Keypoints\n{results["SURF"]["img1_kp"]} points')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cv2.cvtColor(img2_surf, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'SURF on Transformed\n{results["SURF"]["img2_kp"]} points')
        axes[1, 2].axis('off')
    else:
        for i in range(3):
            axes[1, i].text(0.5, 0.5, 'SURF Not Available', 
                          ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].axis('off')
    
    # Row 3: ORB
    if 'ORB' in results:
        axes[2, 0].imshow(img1, cmap='gray')
        axes[2, 0].set_title('Image 1 (Original)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(cv2.cvtColor(img1_orb, cv2.COLOR_BGR2RGB))
        axes[2, 1].set_title(f'ORB Keypoints\n{results["ORB"]["img1_kp"]} points')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(cv2.cvtColor(img2_orb, cv2.COLOR_BGR2RGB))
        axes[2, 2].set_title(f'ORB on Transformed\n{results["ORB"]["img2_kp"]} points')
        axes[2, 2].axis('off')
    else:
        for i in range(3):
            axes[2, i].text(0.5, 0.5, 'ORB Not Available', 
                          ha='center', va='center', transform=axes[2, i].transAxes)
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display comparison table
    print("\nFEATURE DETECTOR COMPARISON:")
    print("=" * 80)
    print(f"{'Method':<10} {'Image1 KP':<12} {'Image2 KP':<12} {'Descriptor Dim':<15} {'Type':<10} {'Repeatability':<15}")
    print("-" * 80)
    
    for method, data in results.items():
        repeatability = data['img2_kp'] / data['img1_kp'] * 100 if data['img1_kp'] > 0 else 0
        print(f"{method:<10} {data['img1_kp']:<12} {data['img2_kp']:<12} "
              f"{data['desc_dim']:<15} {data['desc_type']:<10} {repeatability:<15.1f}%")
    
    # Feature matching demonstration
    print("\nFEATURE MATCHING DEMONSTRATION:")
    print("-" * 40)
    
    def demonstrate_feature_matching(desc1, desc2, kp1, kp2, method_name, matcher_type='bf'):
        if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
            print(f"{method_name}: No features to match")
            return None
        
        # Create matcher based on descriptor type
        if method_name == 'ORB':
            # Binary descriptor → Hamming distance
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            # Floating point descriptor → L2 distance
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # Match descriptors
        matches = matcher.match(desc1, desc2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take top matches
        good_matches = matches[:50]
        
        print(f"{method_name}: {len(good_matches)} good matches found")
        print(f"  Average distance: {np.mean([m.distance for m in good_matches]):.2f}")
        print(f"  Min distance: {min([m.distance for m in good_matches]):.2f}")
        print(f"  Max distance: {max([m.distance for m in good_matches]):.2f}")
        
        return good_matches
    
    # Match features untuk setiap metode
    print("\nMatching Results:")
    if 'SIFT' in results and kp1_sift is not None and kp2_sift is not None:
        matches_sift = demonstrate_feature_matching(desc1_sift, desc2_sift, kp1_sift, kp2_sift, "SIFT")
    
    if 'SURF' in results and kp1_surf is not None and kp2_surf is not None:
        matches_surf = demonstrate_feature_matching(desc1_surf, desc2_surf, kp1_surf, kp2_surf, "SURF")
    
    if 'ORB' in results and kp1_orb is not None and kp2_orb is not None:
        matches_orb = demonstrate_feature_matching(desc1_orb, desc2_orb, kp1_orb, kp2_orb, "ORB")
    
    return results

# Jalankan latihan 1
feature_results = latihan_1()

def latihan_2():
    print("BAG OF VISUAL WORDS AND PCA IMPLEMENTATION")
    print("=" * 50)
    
    # Buat dataset citra sederhana untuk demonstrasi
    def create_simple_dataset():
        images = []
        labels = []
        
        # Class 0: Circles
        for i in range(5):
            img = np.zeros((100, 100), dtype=np.uint8)
            radius = 30 + np.random.randint(-5, 5)
            cv2.circle(img, (50, 50), radius, 255, -1)
            images.append(img)
            labels.append(0)
        
        # Class 1: Squares
        for i in range(5):
            img = np.zeros((100, 100), dtype=np.uint8)
            size = 40 + np.random.randint(-5, 5)
            x = 50 - size//2
            y = 50 - size//2
            cv2.rectangle(img, (x, y), (x+size, y+size), 255, -1)
            images.append(img)
            labels.append(1)
        
        # Class 2: Triangles
        for i in range(5):
            img = np.zeros((100, 100), dtype=np.uint8)
            size = 40 + np.random.randint(-5, 5)
            pts = np.array([[50, 50-size//2], 
                           [50-size//2, 50+size//2], 
                           [50+size//2, 50+size//2]])
            cv2.fillPoly(img, [pts], 255)
            images.append(img)
            labels.append(2)
        
        # Add some noise dan variasi
        for i in range(len(images)):
            noise = np.random.normal(0, 10, images[i].shape)
            images[i] = np.clip(images[i].astype(float) + noise, 0, 255).astype(np.uint8)
        
        return np.array(images), np.array(labels)
    
    images, labels = create_simple_dataset()
    
    # Step 1: Extract features menggunakan ORB (cepat)
    print("\nSTEP 1: FEATURE EXTRACTION")
    print("-" * 30)
    
    orb = cv2.ORB_create(nfeatures=50)
    all_descriptors = []
    all_keypoints = []
    
    for i, img in enumerate(images):
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            all_descriptors.extend(des)
            all_keypoints.append((i, len(kp)))
    
    print(f"Total images: {len(images)}")
    print(f"Total keypoints extracted: {len(all_descriptors)}")
    print(f"Average keypoints per image: {len(all_descriptors)/len(images):.1f}")
    
    # Step 2: Create visual vocabulary menggunakan K-means
    print("\nSTEP 2: VISUAL VOCABULARY CREATION")
    print("-" * 30)
    
    # Convert descriptors to float32 untuk kmeans
    all_descriptors = np.array(all_descriptors, dtype=np.float32)
    
    # Define vocabulary size
    vocab_size = 10  # Small vocabulary untuk demonstrasi
    
    # Kriteria untuk kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    
    # Jalankan kmeans
    print(f"Running K-means clustering with {vocab_size} clusters...")
    _, labels_kmeans, centers = cv2.kmeans(all_descriptors, vocab_size, None, 
                                          criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    print(f"Vocabulary created with {vocab_size} visual words")
    print(f"Visual word dimensions: {centers.shape[1]}")
    
    # Step 3: Quantize features untuk setiap citra
    print("\nSTEP 3: FEATURE QUANTIZATION")
    print("-" * 30)
    
    # Buat matcher untuk quantization
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    image_histograms = []
    
    for i, img in enumerate(images):
        kp, des = orb.detectAndCompute(img, None)
        
        if des is not None:
            # Quantize setiap descriptor ke visual word terdekat
            histogram = np.zeros(vocab_size)
            
            for d in des:
                # Convert ke float32 untuk matching
                d_float = d.astype(np.float32).reshape(1, -1)
                
                # Hitung distance ke setiap visual word
                distances = np.linalg.norm(centers - d_float, axis=1)
                
                # Temukan visual word terdekat
                closest_word = np.argmin(distances)
                histogram[closest_word] += 1
            
            # Normalisasi histogram
            if np.sum(histogram) > 0:
                histogram = histogram / np.sum(histogram)
            
            image_histograms.append(histogram)
        else:
            image_histograms.append(np.zeros(vocab_size))
    
    image_histograms = np.array(image_histograms)
    
    print(f"Created histograms for {len(image_histograms)} images")
    print(f"Histogram shape: {image_histograms.shape}")
    
    # Step 4: TF-IDF Weighting
    print("\nSTEP 4: TF-IDF WEIGHTING")
    print("-" * 30)
    
    def compute_tfidf(histograms):
        # Term Frequency (already normalized)
        tf = histograms
        
        # Document Frequency
        df = np.sum(histograms > 0, axis=0)
        
        # Inverse Document Frequency
        N = histograms.shape[0]
        idf = np.log((N + 1) / (df + 1)) + 1  # Smoothed IDF
        
        # TF-IDF
        tfidf = tf * idf
        
        # Normalize L2
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        tfidf_normalized = tfidf / norms
        
        return tfidf_normalized, tf, idf
    
    tfidf_vectors, tf, idf = compute_tfidf(image_histograms)
    
    print("TF-IDF vectors computed")
    print(f"TF-IDF shape: {tfidf_vectors.shape}")
    
    # Step 5: PCA untuk dimensionality reduction
    print("\nSTEP 5: PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("-" * 30)
    
    def manual_pca(data, n_components=2):
        """Implement PCA manually"""
        # 1. Center data
        mean = np.mean(data, axis=0)
        data_centered = data - mean
        
        # 2. Compute covariance matrix
        covariance_matrix = np.cov(data_centered, rowvar=False)
        
        # 3. Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # 4. Sort eigenvalues dan eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 5. Select top n_components
        components = eigenvectors[:, :n_components]
        
        # 6. Transform data
        transformed = np.dot(data_centered, components)
        
        return transformed, eigenvalues, eigenvectors, mean
    
    # Apply PCA
    pca_result, eigenvalues, eigenvectors, mean = manual_pca(tfidf_vectors, n_components=2)
    
    print(f"Original dimension: {tfidf_vectors.shape[1]}")
    print(f"Reduced dimension: {pca_result.shape[1]}")
    print(f"Eigenvalues: {eigenvalues[:5]}")  # Print top 5 eigenvalues
    
    # Variance explained
    total_variance = np.sum(eigenvalues)
    variance_explained = eigenvalues[:2] / total_variance * 100
    print(f"Variance explained by PC1: {variance_explained[0]:.1f}%")
    print(f"Variance explained by PC2: {variance_explained[1]:.1f}%")
    print(f"Total variance explained: {np.sum(variance_explained):.1f}%")
    
    # Step 6: Visualisasi hasil
    print("\nSTEP 6: VISUALIZATION")
    print("-" * 30)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Sample images
    sample_indices = [0, 5, 10]  # Satu dari setiap kelas
    for idx, sample_idx in enumerate(sample_indices):
        axes[0, idx].imshow(images[sample_idx], cmap='gray')
        class_name = ['Circle', 'Square', 'Triangle'][labels[sample_idx]]
        axes[0, idx].set_title(f'{class_name}\nImage {sample_idx}')
        axes[0, idx].axis('off')
    
    # Plot 2: Visual vocabulary (centers as images)
    axes[0, 2].axis('off')  # Kosongkan plot
    
    # Plot 3: Histogram contoh
    axes[1, 0].bar(range(vocab_size), image_histograms[0])
    axes[1, 0].set_title('Histogram Example (Circle)')
    axes[1, 0].set_xlabel('Visual Word')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: TF-IDF weights untuk visual words
    axes[1, 1].bar(range(vocab_size), idf)
    axes[1, 1].set_title('IDF Weights of Visual Words')
    axes[1, 1].set_xlabel('Visual Word')
    axes[1, 1].set_ylabel('IDF Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: PCA visualization
    colors = ['red', 'blue', 'green']
    for class_idx in range(3):
        class_mask = labels == class_idx
        axes[1, 2].scatter(pca_result[class_mask, 0], pca_result[class_mask, 1], 
                          c=colors[class_idx], label=['Circle', 'Square', 'Triangle'][class_idx],
                          alpha=0.7, s=100)
    
    axes[1, 2].set_title('PCA Visualization of Images')
    axes[1, 2].set_xlabel('Principal Component 1')
    axes[1, 2].set_ylabel('Principal Component 2')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Step 7: Image classification sederhana menggunakan BoVW + PCA
    print("\nSTEP 7: SIMPLE CLASSIFICATION DEMONSTRATION")
    print("-" * 40)
    
    def classify_image_knn(query_hist, training_hists, training_labels, k=3):
        """Simple k-NN classifier menggunakan histogram"""
        # Hitung cosine similarity
        similarities = []
        for train_hist in training_hists:
            # Cosine similarity
            similarity = np.dot(query_hist, train_hist) / (
                np.linalg.norm(query_hist) * np.linalg.norm(train_hist) + 1e-8)
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Ambil k nearest neighbors
        nearest_indices = np.argsort(similarities)[::-1][:k]
        nearest_labels = training_labels[nearest_indices]
        
        # Majority voting
        from collections import Counter
        most_common = Counter(nearest_labels).most_common(1)
        
        return most_common[0][0], similarities[nearest_indices]
    
    # Test dengan beberapa citra
    test_indices = [2, 7, 12]  # Test images
    print("Classification Results:")
    print(f"{'Image':<10} {'True Class':<15} {'Predicted Class':<20} {'Similarity':<10}")
    print("-" * 60)
    
    for test_idx in test_indices:
        # Pisahkan training dan test
        train_mask = np.ones(len(images), dtype=bool)
        train_mask[test_idx] = False
        
        # Train dan test data
        train_hists = tfidf_vectors[train_mask]
        train_labels = labels[train_mask]
        test_hist = tfidf_vectors[test_idx]
        true_label = labels[test_idx]
        
        # Classify
        pred_label, similarities = classify_image_knn(test_hist, train_hists, train_labels, k=3)
        
        true_class = ['Circle', 'Square', 'Triangle'][true_label]
        pred_class = ['Circle', 'Square', 'Triangle'][pred_label]
        
        print(f"{test_idx:<10} {true_class:<15} {pred_class:<20} {np.mean(similarities):<10.3f}")
    
    # Summary
    print("\nBAG OF VISUAL WORDS PIPELINE SUMMARY:")
    print("=" * 50)
    print("1. Feature Extraction: ORB descriptors")
    print("2. Vocabulary Creation: K-means clustering")
    print(f"3. Vocabulary Size: {vocab_size} visual words")
    print("4. Feature Quantization: Nearest visual word assignment")
    print("5. TF-IDF Weighting: Normalized histogram representation")
    print("6. PCA: Dimensionality reduction to 2D")
    print(f"7. Classification: k-NN with cosine similarity")
    print(f"8. Total Images: {len(images)}")
    print(f"9. Classes: {len(np.unique(labels))}")
    
    return {
        'images': images,
        'labels': labels,
        'histograms': image_histograms,
        'tfidf': tfidf_vectors,
        'pca_result': pca_result,
        'vocab_size': vocab_size
    }

# Jalankan latihan 2
bovw_results = latihan_2()