import cv2
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms as transforms
import open3d as o3d
import copy
import glob
from sklearn.neighbors import NearestNeighbors
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
from lib.SuperPoint_V2 import SuperPointFrontend

# ============= NEW ENHANCEMENT CLASSES =============

class TemporalKeypointTracker:
    """Maintains temporal consistency of keypoints across frames"""
    def __init__(self, max_history=5, consistency_threshold=0.7):
        self.keypoint_history = []
        self.descriptor_history = []
        self.max_history = max_history
        self.consistency_threshold = consistency_threshold
        
    def update_and_filter(self, keypoints, descriptors, image_shape):
        """Update keypoint history and apply temporal filtering"""
        temporal_weights = np.ones(len(keypoints))
        
        if len(self.keypoint_history) > 0 and len(keypoints) > 0:
            # Match with previous frame
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            
            # Ensure descriptors are in correct format
            prev_desc = self.descriptor_history[-1].T.astype(np.float32)
            curr_desc = descriptors.T.astype(np.float32)
            
            if prev_desc.shape[0] > 0 and curr_desc.shape[0] > 0:
                matches = bf.match(prev_desc, curr_desc)
                
                # Calculate temporal consistency weights
                for match in matches:
                    # Higher weight for consistently tracked keypoints
                    temporal_weights[match.trainIdx] = min(1.0 + 0.2 * len(self.keypoint_history), 2.0)
                
                # Apply non-maximum suppression with temporal weighting
                keypoints, descriptors = self._weighted_nms(keypoints, descriptors, temporal_weights, image_shape)
        
        # Update history
        self.keypoint_history.append(keypoints)
        self.descriptor_history.append(descriptors)
        
        # Maintain history size
        if len(self.keypoint_history) > self.max_history:
            self.keypoint_history.pop(0)
            self.descriptor_history.pop(0)
            
        return keypoints, descriptors, temporal_weights
    
    def _weighted_nms(self, keypoints, descriptors, weights, image_shape, radius=10):
        """Apply weighted non-maximum suppression"""
        if len(keypoints) == 0:
            return keypoints, descriptors
            
        # Sort by weighted response
        indices = np.argsort(-weights)
        selected = []
        
        h, w = image_shape[:2]
        suppressed = np.zeros((h, w), dtype=bool)
        
        for idx in indices:
            x, y = int(keypoints[idx, 0]), int(keypoints[idx, 1])
            
            if not suppressed[y, x]:
                selected.append(idx)
                # Suppress nearby points
                y_min = max(0, y - radius)
                y_max = min(h, y + radius)
                x_min = max(0, x - radius)
                x_max = min(w, x + radius)
                suppressed[y_min:y_max, x_min:x_max] = True
                
        return keypoints[selected], descriptors[:, selected]

class AdaptivePointCloudProcessor:
    """Handles adaptive point cloud processing with scene-aware parameters"""
    def __init__(self):
        self.scene_complexity = 0.5
        self.last_cloud = None
        
    def estimate_scene_complexity(self, point_cloud, keypoints_count):
        """Estimate scene complexity based on point cloud and keypoints"""
        points = np.asarray(point_cloud.points)
        
        # Factors for complexity
        # 1. Point density variation
        nbrs = NearestNeighbors(n_neighbors=10).fit(points)
        distances, _ = nbrs.kneighbors(points)
        density_var = np.var(distances.mean(axis=1))
        
        # 2. Keypoint density
        keypoint_density = keypoints_count / len(points) if len(points) > 0 else 0
        
        # 3. Spatial distribution (using PCA)
        if len(points) > 100:
            centered = points - points.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            planarity = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2] if eigenvalues[2] > 0 else 0
        else:
            planarity = 0.5
            
        # Combine factors
        self.scene_complexity = 0.3 * density_var + 0.3 * keypoint_density + 0.4 * planarity
        self.scene_complexity = np.clip(self.scene_complexity, 0, 1)
        
        return self.scene_complexity

# ============= ENHANCED DEPTH PROCESSING =============

def compute_depth_confidence(depth_map, rgb_image, window_size=5):
    """Compute confidence map for depth estimation"""
    # Convert to grayscale for gradient computation
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude
    if gradient_mag.max() > 0:
        gradient_confidence = gradient_mag / gradient_mag.max()
    else:
        gradient_confidence = np.zeros_like(gradient_mag)
    
    # Compute depth consistency using Laplacian
    depth_laplacian = cv2.Laplacian(depth_map.astype(np.float32), cv2.CV_64F)
    consistency_confidence = np.exp(-np.abs(depth_laplacian) / (np.abs(depth_laplacian).max() + 1e-6))
    
    # Multi-scale validation
    scales = [1, 2, 4]
    multiscale_confidence = np.zeros_like(depth_map)
    
    for scale in scales:
        # Downsample and upsample
        h, w = depth_map.shape
        scaled_size = (w // scale, h // scale)
        if scaled_size[0] > 0 and scaled_size[1] > 0:
            depth_down = cv2.resize(depth_map, scaled_size, interpolation=cv2.INTER_AREA)
            depth_up = cv2.resize(depth_down, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Compute difference
            scale_error = np.abs(depth_map - depth_up) / (depth_map + 1e-6)
            scale_conf = np.exp(-scale_error)
            multiscale_confidence += scale_conf / len(scales)
    
    # Combine confidence measures
    confidence = (0.4 * gradient_confidence + 
                 0.3 * consistency_confidence + 
                 0.3 * multiscale_confidence)
    
    # Apply Gaussian smoothing to confidence map
    confidence = cv2.GaussianBlur(confidence, (window_size, window_size), 0)
    
    return confidence

def refine_depth_with_confidence(depth_map, confidence, rgb_image, threshold=0.7):
    """Apply confidence-based depth refinement"""
    refined_depth = depth_map.copy().astype(np.float32)
    
    # Create masks for different confidence levels
    high_conf_mask = confidence > threshold
    low_conf_mask = confidence <= threshold
    
    # Apply bilateral filter to high confidence areas
    if np.any(high_conf_mask):
        # Use adaptive parameters based on confidence
        d = 9  # Diameter of pixel neighborhood
        sigma_color = 75
        sigma_space = 75
        
        bilateral_depth = cv2.bilateralFilter(
            refined_depth, d, sigma_color, sigma_space
        )
        refined_depth[high_conf_mask] = bilateral_depth[high_conf_mask]
    
    # Interpolate low confidence areas
    if np.any(low_conf_mask):
        # Create inpainting mask
        inpaint_mask = (low_conf_mask & (depth_map > 0)).astype(np.uint8) * 255
        
        if np.any(inpaint_mask):
            # Use inpainting to fill low confidence regions
            inpainted = cv2.inpaint(
                (refined_depth * 255).astype(np.uint8),
                inpaint_mask,
                5,
                cv2.INPAINT_TELEA
            )
            refined_depth[low_conf_mask] = inpainted[low_conf_mask] / 255.0
    
    # Final edge-preserving smoothing
    refined_depth = cv2.edgePreservingFilter(
        refined_depth, flags=1, sigma_s=60, sigma_r=0.4
    )
    
    return refined_depth

def generate_enhanced_depth_map(rgb_path, depth_model, depth_dir=None):
    """Generate enhanced depth map with confidence-based refinement"""
    # Original depth generation
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        print(f"Error reading image: {rgb_path}")
        return None, None, None, None, None, None
    
    rgb_c = rgb[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (448, 448))
    
    img_torch = scale_torch(A_resize)[None, :, :, :]
    pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
    
    # Compute confidence map
    confidence = compute_depth_confidence(pred_depth_ori, rgb)
    
    # Apply confidence-based refinement
    pred_depth_refined = refine_depth_with_confidence(
        pred_depth_ori, confidence, rgb, threshold=0.7
    )
    
    # Continue with normalization
    pred_depth_normalized = (pred_depth_refined - pred_depth_refined.min()) / (pred_depth_refined.max() - pred_depth_refined.min())
    pred_depth_gray = (pred_depth_normalized * 255).astype(np.uint8)
    pred_depth_eq = cv2.equalizeHist(pred_depth_gray)
    
    # Save enhanced depth map if depth_dir is provided
    if depth_dir is not None:
        filename = os.path.splitext(os.path.basename(rgb_path))[0]
        depth_path = os.path.join(depth_dir, f"{filename}_enhanced.png")
        cv2.imwrite(depth_path, pred_depth_eq)
        
        # Also save confidence map for visualization
        conf_path = os.path.join(depth_dir, f"{filename}_confidence.png")
        cv2.imwrite(conf_path, (confidence * 255).astype(np.uint8))
        print(f"Saved enhanced depth map: {depth_path}")
    
    return rgb, pred_depth_refined, pred_depth_normalized, pred_depth_gray, pred_depth_eq, confidence

# ============= ENHANCED POINT CLOUD PROCESSING =============

def adaptive_knn_smoothing(point_cloud, keypoints_3d=None, base_k=30, edge_threshold=0.3):
    """Apply adaptive KNN smoothing with geometric priors"""
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    
    if len(points) == 0:
        return point_cloud
    
    # Compute normals if not present
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    normals = np.asarray(point_cloud.normals)
    
    # Estimate local density
    nbrs = NearestNeighbors(n_neighbors=base_k).fit(points)
    distances, indices = nbrs.kneighbors(points)
    local_density = 1.0 / (distances.mean(axis=1) + 1e-6)
    mean_density = local_density.mean()
    
    # Compute adaptive k for each point
    adaptive_k = (base_k * (mean_density / local_density) ** 0.5).astype(int)
    adaptive_k = np.clip(adaptive_k, 15, 50)
    
    # Apply adaptive smoothing
    smoothed_points = np.zeros_like(points)
    
    for i in range(len(points)):
        k = adaptive_k[i]
        
        # Find k nearest neighbors
        nbrs_adaptive = NearestNeighbors(n_neighbors=min(k, len(points))).fit(points)
        distances_i, indices_i = nbrs_adaptive.kneighbors([points[i]])
        
        neighbor_points = points[indices_i[0]]
        neighbor_normals = normals[indices_i[0]]
        
        # Compute weights based on distance and normal similarity
        dist_weights = np.exp(-distances_i[0] / (distances_i[0].mean() + 1e-6))
        
        # Normal similarity weights
        normal_similarity = np.abs(np.dot(neighbor_normals, normals[i]))
        normal_weights = np.exp((normal_similarity - 1) / 0.2)
        
        # Combined weights
        weights = dist_weights * normal_weights
        weights = weights / weights.sum()
        
        # Weighted average position
        smoothed_points[i] = np.sum(neighbor_points * weights[:, np.newaxis], axis=0)
        
        # Edge preservation
        if is_edge_point(points[i], neighbor_points, neighbor_normals, edge_threshold):
            # Reduce smoothing for edge points
            lambda_edge = 0.3
            smoothed_points[i] = lambda_edge * points[i] + (1 - lambda_edge) * smoothed_points[i]
    
    # Create smoothed point cloud
    smoothed_pcd = o3d.geometry.PointCloud()
    smoothed_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
    smoothed_pcd.colors = o3d.utility.Vector3dVector(colors)
    smoothed_pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return smoothed_pcd

def is_edge_point(point, neighbors, neighbor_normals, threshold=0.3):
    """Detect if a point is on an edge based on normal variation"""
    if len(neighbor_normals) < 3:
        return False
    
    # Compute normal variation
    normal_var = np.var(neighbor_normals, axis=0).sum()
    
    return normal_var > threshold

def multi_stage_filtering(point_cloud, scene_complexity=0.5):
    """Apply multi-stage filtering based on scene complexity"""
    filtered_pcd = copy.deepcopy(point_cloud)
    
    # Stage 1: Statistical outlier removal
    k_neighbors = int(30 * (1 + scene_complexity))
    std_ratio = 2.0 - 0.5 * scene_complexity  # Less aggressive for complex scenes
    
    filtered_pcd, ind = filtered_pcd.remove_statistical_outlier(
        nb_neighbors=k_neighbors,
        std_ratio=std_ratio
    )
    
    # Stage 2: Radius outlier removal for isolated points
    radius = 0.05 * (1 + scene_complexity)
    min_points = int(10 * (1 + scene_complexity))
    
    filtered_pcd, ind = filtered_pcd.remove_radius_outlier(
        nb_points=min_points,
        radius=radius
    )
    
    # Stage 3: Voxel downsampling if needed
    if len(filtered_pcd.points) > 100000:
        voxel_size = 0.02 * (1 + 0.5 * scene_complexity)
        filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    return filtered_pcd

# ============= ENHANCED MOTION ESTIMATION =============

def hybrid_motion_estimation(source_pcd, target_pcd, scene_complexity=0.5, initial_transform=np.eye(4)):
    """Hybrid ICP-NDT motion estimation with scene-adaptive fusion"""
    
    # Ensure point clouds have normals
    if not source_pcd.has_normals():
        source_pcd.estimate_normals()
    if not target_pcd.has_normals():
        target_pcd.estimate_normals()
    
    # RANSAC-based initial alignment
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd,
        o3d.pipelines.registration.CorrespondenceSet(),  # Will be computed internally
        max_correspondence_distance=0.05,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    # ICP refinement
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=0.02,
        init=ransac_result.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    # For NDT, we'll simulate it with a different ICP variant
    # (Open3D doesn't have built-in NDT, but we can approximate)
    ndt_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=0.05,
        init=ransac_result.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )
    
    # Adaptive fusion based on scene complexity
    # Higher complexity -> more weight to NDT (more robust to local minima)
    # Lower complexity -> more weight to ICP (more precise)
    lambda_icp = 1.0 - scene_complexity
    lambda_ndt = scene_complexity
    
    # Fuse transformations (simplified - in practice, you'd fuse SE(3) properly)
    T_hybrid = lambda_icp * icp_result.transformation + lambda_ndt * ndt_result.transformation
    
    # Validate transformation
    if is_transformation_valid(T_hybrid):
        return T_hybrid, icp_result.fitness
    else:
        # Fallback to best individual result
        if is_transformation_valid(icp_result.transformation):
            return icp_result.transformation, icp_result.fitness
        elif is_transformation_valid(ndt_result.transformation):
            return ndt_result.transformation, ndt_result.fitness
        else:
            print("Warning: No valid transformation found, using identity")
            return np.eye(4), 0.0

def is_transformation_valid(transform, rotation_threshold=0.1, translation_threshold=5.0):
    """Validate if transformation is physically plausible"""
    # Check rotation validity
    R = transform[:3, :3]
    det = np.linalg.det(R)
    
    if abs(det - 1.0) > rotation_threshold:
        return False
    
    # Check if rotation matrix is orthogonal
    should_be_identity = np.dot(R, R.T)
    if np.linalg.norm(should_be_identity - np.eye(3)) > rotation_threshold:
        return False
    
    # Check translation magnitude
    t = transform[:3, 3]
    if np.linalg.norm(t) > translation_threshold:
        return False
    
    return True

# ============= MODIFIED MAIN PROCESSING FUNCTIONS =============

def process_single_image_enhanced(image_path, depth_model, args, point_cloud_dir=None, 
                                 keypoints_dir=None, depth_dir=None, 
                                 temporal_tracker=None, scene_processor=None):
    """Enhanced single image processing with all improvements"""
    
    filename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing image with enhancements: {image_path}")
    
    # Generate enhanced depth maps with confidence
    rgb, pred_depth_refined, pred_depth_normalized, pred_depth_gray, pred_depth_eq, confidence = \
        generate_enhanced_depth_map(image_path, depth_model, depth_dir)
    
    if rgb is None:
        print(f"Skipping {image_path} due to reading error")
        return None
    
    # Keypoint detection with temporal consistency
    keypoints_2d = None
    descriptors = None
    
    if args.use_keypoints and args.keypoint_weight:
        try:
            # Detect keypoints
            pts, desc = process_image(
                image_path,
                args.keypoint_weight,
                args.nms_dist,
                args.conf_thresh,
                args.cuda,
                args.display_scale,
                None  # Don't save yet
            )
            
            # Apply temporal consistency if tracker is available
            if temporal_tracker is not None:
                keypoints_2d = pts[:2, :].T
                descriptors = desc
                
                keypoints_2d, descriptors, temporal_weights = temporal_tracker.update_and_filter(
                    keypoints_2d, descriptors, rgb.shape
                )
                
                print(f"Detected {len(keypoints_2d)} temporally consistent keypoints")
            else:
                keypoints_2d = pts[:2, :].T
                descriptors = desc
                print(f"Detected {pts.shape[1]} keypoints")
                
            # Save keypoints if directory provided
            if keypoints_dir is not None:
                keypoints_path = os.path.join(keypoints_dir, f"{filename}_enhanced.npy")
                np.save(keypoints_path, keypoints_2d)
                
        except Exception as e:
            print(f"Error detecting keypoints: {e}")
            keypoints_2d = None
    
    # Generate point cloud with refined depth
    pcd = depth_to_pointcloud(
        rgb, pred_depth_eq,
        fx=args.fx, fy=args.fy,
        cx=args.cx, cy=args.cy,
        depth_scale=args.depth_scale,
        skip=args.skip
    )
    
    # Estimate scene complexity
    if scene_processor is not None:
        scene_complexity = scene_processor.estimate_scene_complexity(
            pcd, len(keypoints_2d) if keypoints_2d is not None else 0
        )
        print(f"Scene complexity: {scene_complexity:.2f}")
    else:
        scene_complexity = 0.5
    
    # Apply multi-stage filtering
    filtered_pcd = multi_stage_filtering(pcd, scene_complexity)
    print(f"Points after multi-stage filtering: {len(filtered_pcd.points)}")
    
    # Apply adaptive smoothing
    if args.smooth:
        if args.use_keypoints and keypoints_2d is not None:
            print("Applying adaptive KNN smoothing")
            result_cloud = adaptive_knn_smoothing(filtered_pcd, keypoints_2d, base_k=30)
        else:
            print("Applying standard smoothing")
            result_cloud, _ = smoothness(
                rgb=rgb,
                depth_map=pred_depth_eq,
                point_cloud=filtered_pcd,
                keypoints=None,
                distance_threshold=0.03,
                ransac_n=3,
                num_iterations=1000,
                show_difference=args.show_difference,
                no_display=args.no_display
            )
    else:
        result_cloud = filtered_pcd
    
    # Visualization
    if not args.no_display:
        # Show confidence map
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(rgb[:,:,::-1])
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(pred_depth_eq, cmap='gray')
        plt.title("Enhanced Depth")
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(confidence, cmap='hot')
        plt.title("Confidence Map")
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show point cloud
        o3d.visualization.draw_geometries([result_cloud], 
                                        window_name=f"Enhanced Point Cloud - {filename}")
    
    # Save enhanced point cloud
    if point_cloud_dir is not None:
        suffix = "_enhanced_smooth" if args.smooth else "_enhanced"
        output_path = os.path.join(point_cloud_dir, f"{filename}{suffix}.ply")
        o3d.io.write_point_cloud(output_path, result_cloud)
        print(f"Saved enhanced point cloud to {output_path}")
    
    # Store for motion estimation
    if scene_processor is not None:
        scene_processor.last_cloud = result_cloud
    
    return result_cloud

def run_enhanced_visual_odometry(image_paths, depth_model, args):
    """Run enhanced visual odometry with all improvements"""
    
    # Initialize components
    temporal_tracker = TemporalKeypointTracker(max_history=5)
    scene_processor = AdaptivePointCloudProcessor()
    
    # Initialize trajectory
    trajectory = []
    current_pose = np.eye(4)
    trajectory.append(current_pose[:3, 3])
    
    # Storage for point clouds
    previous_cloud = None
    
    # Create output directories
    if args.output_dir:
        depth_dir = os.path.join(args.output_dir, "enhanced_depth")
        keypoints_dir = os.path.join(args.output_dir, "enhanced_keypoints")
        point_cloud_dir = os.path.join(args.output_dir, "enhanced_point_clouds")
        
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(keypoints_dir, exist_ok=True)
        os.makedirs(point_cloud_dir, exist_ok=True)
    else:
        depth_dir = keypoints_dir = point_cloud_dir = None
    
    # Process each frame
    for i, image_path in enumerate(image_paths):
        print(f"\n--- Processing frame {i+1}/{len(image_paths)} ---")
        
        # Process frame with enhancements
        current_cloud = process_single_image_enhanced(
            image_path, depth_model, args,
            point_cloud_dir, keypoints_dir, depth_dir,
            temporal_tracker, scene_processor
        )
        
        if current_cloud is None:
            continue
        
        # Motion estimation for frame pairs
        if previous_cloud is not None and i > 0:
            print("Estimating motion...")
            
            # Get scene complexity from processor
            scene_complexity = scene_processor.scene_complexity
            
            # Hybrid motion estimation
            transform, fitness = hybrid_motion_estimation(
                current_cloud, previous_cloud, scene_complexity, np.eye(4)
            )
            
            print(f"Motion estimation fitness: {fitness:.4f}")
            
            # Update pose
            current_pose = current_pose @ transform
            trajectory.append(current_pose[:3, 3].copy())
            
            # Print motion statistics
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            angle = np.arccos((np.trace(rotation) - 1) / 2) * 180 / np.pi
            print(f"Rotation: {angle:.2f} degrees, Translation: {np.linalg.norm(translation):.3f} meters")
        
        previous_cloud = current_cloud
    
    # Plot enhanced trajectory
    trajectory = np.array(trajectory)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(trajectory[:, 0], trajectory[:, 2], 'b-o', markersize=4)
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.title('Enhanced Visual Odometry Trajectory (X-Z plane)')
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(212)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', markersize=4)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Enhanced Visual Odometry Trajectory (X-Y plane)')
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return trajectory

# ============= MODIFIED MAIN FUNCTION =============

def main():
    args = parse_args()
    
    # Add new argument for enhanced processing
    args.enhanced = True  # Set this to True to use enhanced processing
    
    # Check if input is a file or directory
    is_file = os.path.isfile(args.input)
    is_dir = os.path.isdir(args.input)
    
    if not (is_file or is_dir):
        print(f"Error: {args.input} is neither a valid file nor a directory")
        return
    
    # Initialize depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()
    load_ckpt(args, depth_model, None, None)
    depth_model.to(device='cpu')
    
    if args.enhanced and args.run_vo and is_dir:
        # Run enhanced visual odometry on directory
        print("Running ENHANCED visual odometry pipeline...")
        
        # Get list of images
        image_paths = sorted(glob.glob(os.path.join(args.input, f"*.{args.file_extension}")))
        print(f"Found {len(image_paths)} images for enhanced processing")
        
        # Run enhanced VO
        trajectory = run_enhanced_visual_odometry(image_paths, depth_model, args)
        
    else:
        # Original processing (fallback)
        print("Running standard processing...")
        # ... (rest of original main function)
    
    print("Enhanced processing complete!")
def parse_args():
    parser = argparse.ArgumentParser(description='Unified Visual Odometry and Point Cloud Generation System')
    # Input/output options
    parser.add_argument('--input', required=True, help='Input RGB image file or directory containing RGB images')
    parser.add_argument('--output_dir', default='./output', help='Output directory for results')
    parser.add_argument('--file_extension', type=str, default="jpg", help="Image file extension (default: jpg)")
    parser.add_argument('--poses_file', help='Path to ground truth poses file (optional)')
    
    # Depth model options
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load depth model')
    parser.add_argument('--backbone', default='resnext101', help='Backbone model type')
    parser.add_argument('--depth_scale', type=float, default=1000.0, 
                       help='Scale factor to convert depth values to meters (default: 1000.0)')
    
    # Camera parameters
    parser.add_argument('--fx', type=float, help='Focal length x (optional)')
    parser.add_argument('--fy', type=float, help='Focal length y (optional)')
    parser.add_argument('--cx', type=float, help='Principal point x (optional)')
    parser.add_argument('--cy', type=float, help='Principal point y (optional)')
    
    # Point cloud options
    parser.add_argument('--skip', type=int, default=1, help='Sample every nth pixel for point cloud (default: 1)')
    parser.add_argument('--smooth', action='store_true', help="Use Smoothing algorithm on the generated point cloud")
    parser.add_argument('--show_difference', action='store_true', help="Show the outliers")
    
    # SuperPoint options
    parser.add_argument("--keypoint_weight", type=str, default="./superpoint_v1.pth", 
                        help="Path to pretrained SuperPoint weights file")
    parser.add_argument("--nms_dist", type=int, default=4, help="Non Maximum Suppression (NMS) distance (default: 4)")
    parser.add_argument("--conf_thresh", type=float, default=0.015, help="Detector confidence threshold (default: 0.015)")
    parser.add_argument("--cuda", action="store_true", help="Use cuda GPU to speed up processing (default: False)")
    
    # Visualization options
    parser.add_argument('--no_display', action='store_true', help="Don't display visualizations")
    parser.add_argument("--display_scale", type=int, default=2, help="Factor to scale output visualization (default: 2)")
    parser.add_argument("--use_keypoints", action='store_true', help="Smooth using keypoints")
    
    # Visual Odometry options
    parser.add_argument('--run_vo', action='store_true', help="Run visual odometry on the image sequence")
    
    args = parser.parse_args()
    return args

def scale_torch(img):
    """Scale the image and output it in torch.tensor."""
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

def generate_depth_map(rgb_path, depth_model, depth_dir=None):
    """Generate depth map from RGB image."""
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        print(f"Error reading image: {rgb_path}")
        return None, None, None, None, None
    
    rgb_c = rgb[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (448, 448))
    
    img_torch = scale_torch(A_resize)[None, :, :, :]
    pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
    
    # Normalize depth map
    pred_depth_normalized = (pred_depth_ori - pred_depth_ori.min()) / (pred_depth_ori.max() - pred_depth_ori.min())
    pred_depth_gray = (pred_depth_normalized * 255).astype(np.uint8)
    pred_depth_eq = cv2.equalizeHist(pred_depth_gray)
    
    # Save depth map if depth_dir is provided
    if depth_dir is not None:
        filename = os.path.splitext(os.path.basename(rgb_path))[0]
        depth_path = os.path.join(depth_dir, f"{filename}.png")
        cv2.imwrite(depth_path, pred_depth_eq)
        print(f"Saved depth map: {depth_path}")
    
    return rgb, pred_depth_ori, pred_depth_normalized, pred_depth_gray, pred_depth_eq

def depth_to_pointcloud(rgb, depth, fx=None, fy=None, cx=None, cy=None, depth_scale=1000.0, skip=1):
    """Convert depth map to point cloud."""
    height, width = depth.shape
    
    # Set default camera parameters if not provided
    if fx is None or fy is None:
        fx = fy = max(width, height) * 0.8
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(0, width, skip), np.arange(0, height, skip))
    x = x.flatten()
    y = y.flatten()
    
    # Convert depth to meters and filter invalid values
    z = depth[y, x].astype(np.float32) / depth_scale
    valid_mask = (z > 0) & np.isfinite(z)
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

    # Convert to 3D coordinates
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    z_3d = z

    # Get colors
    colors = rgb[y, x] / 255.0

    # Create point cloud
    points = np.stack([x_3d, y_3d, z_3d], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def process_image(image_path, weights_path, nms_dist=4, conf_thresh=0.015, cuda=False, display_scale=2, keypoints_dir=None):
    """Process a single image with SuperPoint."""
    # Read and preprocess image
    img = cv2.imread(image_path, 0)
    if img is None:
        raise Exception(f"Error reading image {image_path}")
    img = img.astype("float32") / 255.0

    # Initialize SuperPoint
    fe = SuperPointFrontend(
        weights_path=weights_path, nms_dist=nms_dist, conf_thresh=conf_thresh, cuda=cuda
    )

    # Process image
    pts, desc, heatmap = fe.run(img)
    
    # Save keypoints if keypoints_dir is provided
    if keypoints_dir is not None:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        keypoints_path = os.path.join(keypoints_dir, f"{filename}.npy")
        np.save(keypoints_path, pts[:2, :].T)
        print(f"Saved keypoints: {keypoints_path}")

    return pts, desc

def smoothness(rgb, depth_map, point_cloud, keypoints, distance_threshold=0.01, ransac_n=5, num_iterations=1000, show_difference=False, no_display=False):
    
    if keypoints is None:
        # Fit a plane model using RANSAC
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        inlier_cloud = point_cloud.select_by_index(inliers)  
        outlier_cloud = point_cloud.select_by_index(inliers, invert=True)  # Outliers

        print("Number of inliers (cleaned points):", len(inlier_cloud.points))
        print("Number of outliers (removed points):", len(outlier_cloud.points))
        
        if show_difference and not no_display:
            i_cloud = copy.deepcopy(inlier_cloud)
            o_cloud = copy.deepcopy(outlier_cloud)
            
            i_cloud.paint_uniform_color([0, 1, 0])
            o_cloud.paint_uniform_color([1, 0, 0])
            
            o3d.visualization.draw_geometries([i_cloud, o_cloud], window_name="Inliers (Green) and Outliers (Red)")
            
        if not no_display:
            o3d.visualization.draw_geometries([inlier_cloud], window_name="After RANSAC Smoothness")
        
        return inlier_cloud, outlier_cloud
    
    else:
        def lift_keypoints_to_3d(keypoints_2d, depth_map, K):
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            keypoints_3d = []
            h, w = depth_map.shape

            for (u, v) in keypoints_2d:
                u_int, v_int = int(round(u)), int(round(v))
                if 0 <= u_int < w and 0 <= v_int < h:
                    z = depth_map[v_int, u_int]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    keypoints_3d.append([x, y, z])

            return np.array(keypoints_3d, dtype=np.float32)
        
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        
        depth_map = depth_map.astype(np.float32)
        
        height, width = depth_map.shape
        
        fx = fy = max(width, height) * 0.8
        cx = width / 2.0
        cy = height / 2.0

        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0,  1 ]])
        
        keypoints_2d = keypoints
        
        keypoints_3d = lift_keypoints_to_3d(keypoints_2d, depth_map, K)
        
        knn = NearestNeighbors(n_neighbors=16, algorithm='auto')
        knn.fit(points)  # Fit KNN on the point cloud

        # Find neighbors for each point
        distances, indices = knn.kneighbors(points)

        # Enhance point cloud by averaging neighbor positions
        enhanced_points = np.copy(points)
        for i in range(points.shape[0]):
            neighbor_pts = points[indices[i]]
            enhanced_points[i] = np.mean(neighbor_pts, axis=0)  # Smooth position

        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(points)
        pcd_original.colors = o3d.utility.Vector3dVector(colors)

        pcd_enhanced = o3d.geometry.PointCloud()
        pcd_enhanced.points = o3d.utility.Vector3dVector(enhanced_points)
        pcd_enhanced.colors = o3d.utility.Vector3dVector(colors) 
        
        if show_difference and not no_display:
            orig = copy.deepcopy(pcd_original).paint_uniform_color([1, 0, 0])  # Red color for original
            enhan = copy.deepcopy(pcd_enhanced).paint_uniform_color([0, 1, 0])  # Green color for enhanced

            o3d.visualization.draw_geometries([orig, enhan], window_name="Original (Red) vs Enhanced (Green)")
        
        if not no_display:
            o3d.visualization.draw_geometries([pcd_enhanced], window_name="Point Cloud after Smoothness")
        
        return pcd_original, pcd_enhanced
    
def visualize_depth_maps(rgb, pred_depth_ori, pred_depth_normalized, pred_depth_gray, pred_depth_eq):
    """Visualize different versions of depth maps."""
    fig = plt.figure(figsize=(8, 10))
    
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])  # First row taller

    # Top image spanning 2 columns
    ax1 = plt.subplot(gs[0, :])  # Row 0, span all columns
    ax1.imshow(rgb)
    ax1.set_title("Image")
    ax1.axis('off')
    
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    ax4 = plt.subplot(gs[2, 0])
    ax5 = plt.subplot(gs[2, 1])

    im = ax2.imshow(pred_depth_ori, cmap='rainbow')
    ax2.set_title('Original Depth Map')
    ax2.axis('off')
    fig.colorbar(im, ax=ax2)
    
    im = ax3.imshow(pred_depth_normalized, cmap='rainbow')
    ax3.set_title('Normalized Depth Map')
    ax3.axis('off')
    fig.colorbar(im, ax=ax3)
    
    im = ax4.imshow(pred_depth_gray, cmap='gray')
    ax4.set_title('Grayscale Depth Map')
    ax4.axis('off')
    fig.colorbar(im, ax=ax4)
    
    im = ax5.imshow(pred_depth_eq, cmap='gray')
    ax5.set_title('Equalized Depth Map')
    ax5.axis('off')
    fig.colorbar(im, ax=ax5)

    # Show the figure
    plt.tight_layout()
    plt.show()

def process_single_image(image_path, depth_model, args, point_cloud_dir=None, keypoints_dir=None, depth_dir=None):
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"Processing image: {image_path}")
    
    # Generate depth maps
    rgb, pred_depth_ori, pred_depth_normalized, pred_depth_gray, pred_depth_eq = generate_depth_map(image_path, depth_model, depth_dir)
    
    if rgb is None:
        print(f"Skipping {image_path} due to reading error")
        return None
    
    keypoints_2d = None
    if args.use_keypoints and args.keypoint_weight:
        try:
            # keypoint detector
            pts, desc = process_image(
                image_path,
                args.keypoint_weight,
                args.nms_dist,
                args.conf_thresh,
                args.cuda,
                args.display_scale,
                keypoints_dir
            )
            keypoints_2d = pts[:2, :].T
            print(f"Detected {pts.shape[1]} keypoints")
        except Exception as e:
            print(f"Error detecting keypoints: {e}")
            keypoints_2d = None
    
    # Generate point cloud
    pcd = depth_to_pointcloud(
        rgb, pred_depth_eq,
        fx=args.fx, fy=args.fy,
        cx=args.cx, cy=args.cy,
        depth_scale=args.depth_scale,
        skip=args.skip
    )
    
    if not args.no_display:
        visualize_depth_maps(rgb, pred_depth_ori, pred_depth_normalized, pred_depth_gray, pred_depth_eq)
        o3d.visualization.draw_geometries([pcd], window_name=f"Original Point Cloud - {filename}")
        
    result_cloud = pcd  # Default result is the original point cloud
    
    if args.smooth:
        if args.use_keypoints and keypoints_2d is not None:
            print("Smoothing Using KNN")
            _, enhanced = smoothness(
                rgb=rgb,
                depth_map=pred_depth_eq,
                point_cloud=pcd,
                keypoints=keypoints_2d,
                distance_threshold=0.03,
                ransac_n=3,
                num_iterations=1000,
                show_difference=args.show_difference,
                no_display=args.no_display)
            
            result_cloud = enhanced
        else:
            print("Smoothing Using RANSAC")
            enhanced, _ = smoothness(
                rgb=rgb,
                depth_map=pred_depth_eq,
                point_cloud=pcd,
                keypoints=None,
                distance_threshold=0.03,
                ransac_n=3,
                num_iterations=1000,
                show_difference=args.show_difference,
                no_display=args.no_display)
            
            result_cloud = enhanced
    
    # Save point cloud if directory is provided
    if point_cloud_dir is not None:
        suffix = "_smooth" if args.smooth else ""
        output_path = os.path.join(point_cloud_dir, f"{filename}{suffix}.ply")
        o3d.io.write_point_cloud(output_path, result_cloud)
        print(f"Saved point cloud to {output_path}")
    
    return result_cloud

def superpoint_detect(image, weights_path, nms_dist=4, conf_thresh=0.015, cuda=False):
    """Detect keypoints and descriptors using SuperPoint."""
    # Convert image to grayscale and normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32) / 255.0

    # Initialize SuperPoint
    fe = SuperPointFrontend(
        weights_path=weights_path, nms_dist=nms_dist, conf_thresh=conf_thresh, cuda=cuda
    )

    # Forward pass through SuperPoint
    with torch.no_grad():
        pts, desc, heatmap = fe.run(image)
        
    keypoints = pts[:2, :].T
    descriptors = desc

    return keypoints, descriptors.T  # Transpose descriptors to (N, 256)

def run_visual_odometry(image_paths, depth_paths, poses_file, fx, fy, cx, cy, keypoint_weight, nms_dist=4, conf_thresh=0.015, cuda=False):
    """Run visual odometry on the image sequence."""
    # Initialize trajectory and pose
    trajectory = []
    current_pose = np.eye(4)  # Initial pose at origin
    trajectory.append(current_pose[:3, 3])  # Save initial position

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Load ground truth poses if available
    gt_poses = []
    if os.path.exists(poses_file):
        with open(poses_file, "r") as file:
            for line in file:
                # Read 12 values (3x3 rotation + 1x3 translation)
                values = list(map(float, line.strip().split()))
                R = np.array(values[:9]).reshape(3, 3)  # 3x3 rotation matrix
                t = np.array(values[9:12])  # 1x3 translation vector
                # Create 4x4 transformation matrix
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                gt_poses.append(T)
    else:
        print("Ground truth poses file not found. Skipping ground truth trajectory.")

    # Process frame pairs
    for i in range(len(image_paths) - 1):
        # Load frames and depth maps
        prev_img = cv2.imread(image_paths[i])
        curr_img = cv2.imread(image_paths[i + 1])
        prev_depth = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)
        
        height, width = prev_depth.shape
        
        fx = fy = max(width, height) * 0.8
        cx = width / 2.0
        cy = height / 2.0
        
        # Define camera intrinsic matrix K
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
        
        prev_depth = prev_depth.astype(np.float32) / 1000.0  # Convert to meters

        # Detect keypoints and descriptors using SuperPoint
        kp1, des1 = superpoint_detect(prev_img, keypoint_weight, nms_dist, conf_thresh, cuda)
        kp2, des2 = superpoint_detect(curr_img, keypoint_weight, nms_dist, conf_thresh, cuda)

        # Match keypoints between frames using BFMatcher
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            # Convert descriptors to CV_32F (required by BFMatcher with L2 norm)
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
            
            # Perform matching
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:100]  # Keep top 100 matches
        else:
            print(f"Frame {i}: No descriptors found. Skipping frame pair.")
            continue

        # Extract 2D points and filter valid depth
        prev_pts = np.array([kp1[m.queryIdx] for m in matches])
        curr_pts = np.array([kp2[m.trainIdx] for m in matches])
        
        # Convert to 3D using depth (prev frame)
        valid_depth_mask = prev_depth[prev_pts[:, 1].astype(int), prev_pts[:, 0].astype(int)] > 0
        prev_pts = prev_pts[valid_depth_mask]
        curr_pts = curr_pts[valid_depth_mask]
        
        if len(prev_pts) < 4:
            print(f"Frame {i}: Not enough valid matches. Skipping frame pair.")
            continue

        z = prev_depth[prev_pts[:, 1].astype(int), prev_pts[:, 0].astype(int)]
        x = (prev_pts[:, 0] - cx) * z / fx
        y = (prev_pts[:, 1] - cy) * z / fy
        pts_3d = np.vstack((x, y, z)).T

        # Solve PnP with RANSAC
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.reshape(-1, 1, 3), 
            curr_pts.reshape(-1, 1, 2), 
            K, None, flags=cv2.SOLVEPNP_ITERATIVE, 
            iterationsCount=100, reprojectionError=3.0
        )

        if not ret:
            print(f"Frame {i}: PnP failed. Skipping frame pair.")
            continue

        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Update pose: current_pose = previous_pose * inv([R|t])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        current_pose = current_pose @ np.linalg.inv(T)
        
        # Save translation part of the pose
        trajectory.append(current_pose[:3, 3].copy())

    # Convert predicted trajectory to numpy array
    pred_trajectory = np.array(trajectory)

    # Plot trajectory
    plt.figure(figsize=(10, 6))
    if gt_poses:
        # Align predicted trajectory with GT initial pose
        T_gt_initial = gt_poses[0]
        T_gt_initial_inv = np.linalg.inv(T_gt_initial)
        pred_trajectory_aligned = []
        for pose in trajectory:
            T_pred = np.eye(4)
            T_pred[:3, 3] = pose
            T_pred_aligned = T_gt_initial_inv @ T_pred  # Transform to GT initial coordinate system
            pred_trajectory_aligned.append(T_pred_aligned[:3, 3])
        pred_trajectory_aligned = np.array(pred_trajectory_aligned)

        # Extract GT trajectory (X, Y, Z coordinates)
        gt_trajectory = np.array([pose[:3, 3] for pose in gt_poses])

        # Plot GT and aligned predicted trajectories (X-Z plane)
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], label='Ground Truth', color='blue', marker='o')
        plt.plot(pred_trajectory_aligned[:, 0], pred_trajectory_aligned[:, 2], label='Predicted (Aligned)', color='red', marker='x')
    else:
        # Plot only predicted trajectory (X-Z plane)
        plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], label='Predicted', color='red', marker='x')

    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.title('Trajectory')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    args = parse_args()
    
    # Check if input is a file or directory
    is_file = os.path.isfile(args.input)
    is_dir = os.path.isdir(args.input)
    
    if not (is_file or is_dir):
        print(f"Error: {args.input} is neither a valid file nor a directory")
        return
    
    # Initialize depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()
    load_ckpt(args, depth_model, None, None)
    depth_model.to(device='cpu')
    
    if is_file:
        # Single file processing mode
        print(f"Processing single file: {args.input}")
        
        # If output directory is provided, use it for saving results
        point_cloud_dir = None
        depth_dir = None
        keypoints_dir = None
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            point_cloud_dir = os.path.join(args.output_dir, "point_clouds")
            depth_dir = os.path.join(args.output_dir, "depth")
            keypoints_dir = os.path.join(args.output_dir, "keypoints")
            
            os.makedirs(point_cloud_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(keypoints_dir, exist_ok=True)
        
        # Process the image
        process_single_image(args.input, depth_model, args, point_cloud_dir, keypoints_dir, depth_dir)
        
    else:
        # Directory processing mode
        print(f"Processing directory: {args.input}")
        
        # Check if output directory is provided
        if not args.output_dir:
            print("Error: --output_dir is required when processing a directory")
            return
        
        # Create output directories
        depth_dir = os.path.join(args.output_dir, "depth")
        keypoints_dir = os.path.join(args.output_dir, "keypoints")
        point_cloud_dir = os.path.join(args.output_dir, "point_clouds")
        
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(keypoints_dir, exist_ok=True)
        os.makedirs(point_cloud_dir, exist_ok=True)
        
        # Get list of images in input directory
        image_paths = glob.glob(os.path.join(args.input, f"*.{args.file_extension.lower()}"))
        image_paths.extend(glob.glob(os.path.join(args.input, f"*.{args.file_extension.upper()}")))
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process each image
        for image_path in image_paths:
            process_single_image(image_path, depth_model, args, point_cloud_dir, keypoints_dir, depth_dir)
        
        # Run visual odometry if requested
        if args.run_vo:
            # Get depth paths
            depth_paths = [os.path.join(depth_dir, (os.path.splitext(os.path.basename(image_path))[0] + ".png" for image_path in image_paths))]
            
            # Run visual odometry
            run_visual_odometry(
                image_paths, depth_paths, args.poses_file,
                args.fx, args.fy, args.cx, args.cy,
                args.keypoint_weight, args.nms_dist, args.conf_thresh, args.cuda
            )
    
    print("Processing complete!")

if __name__ == '__main__':
    main()
# Keep all original functions (parse_args, scale_torch, etc.) as they are
# The enhanced versions can coexist with the original ones