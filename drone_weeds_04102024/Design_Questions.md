# Design Questions

## 1

### 1a

A suitable deep learning architecture for real-time weed detection could be based on a Convolutional Neural Network (CNN) backbone integrated with a real-time object detection model like YOLOv8 (You Only Look Once) or EfficientDet. These architectures are designed for high accuracy while maintaining low inference latency, making them ideal for real-time tasks on drones.

* Input: High-resolution drone imagery
* Backbone: A CNN (e.g., MobileNetV3, ResNet50) to extract feature maps from the images.
* Neck: Feature pyramid network (FPN) to ensure detection at multiple scales.
* Head: YOLOv8 or EfficientDet’s detection head for bounding box regression and weed classification.

This architecture is suitable because:
* YOLO models balance speed and accuracy, making them well-suited for edge devices like drones.
* EfficientDet offers scalability and efficiency for different computational budgets.

### 1b

To balance accuracy with the need for real-time performance:
* Model Pruning and Quantization
    * Techniques like weight pruning (removing unnecessary connections) and quantization (reducing the precision of the model parameters) can significantly reduce the model size and speed up inference without a large impact on accuracy.
* Smaller Backbone
    * Using a lightweight backbone like MobileNetV3 instead of heavier architectures can help maintain real-time performance while preserving reasonable accuracy.
* Resolution Downscaling
    * Lower the input resolution during inference without drastically affecting the weed detection quality.
* Hardware Optimization
    * Utilize GPU or TPU acceleration on the drone hardware (if available) to speed up inference. Techniques like FP16 precision can further boost performance.

### 1c

* Data Augmentation
    * Lighting and Color Jittering: Randomly adjust the brightness, contrast, saturation, and hue of images during training to simulate different lighting conditions.
* Random Rotations and Flips: 
    * Introduce variations in angle through random rotations, flipping, and perspective warping.
* Motion Blur Simulation
    * Add motion blur to the training data to simulate movement of the drone during capture.
* Domain Adaptation
    * Train the model on images captured under different conditions (e.g., sunny, cloudy, low light) to ensure generalization across environments.
* Temporal Smoothing
    * Use techniques like temporal averaging or tracking to smooth predictions over consecutive frames to account for motion-related errors.
* Occlusion Handling
    * Employ CutMix or Occlusion Augmentation to simulate occluded objects in the training set, teaching the model to handle partially visible weeds.

    ## What metrics would you use to assess the performance of this model, given the initially broad definition of the system?

### 1d
* Mean Average Precision (mAP)
    * This measures the precision and recall across different intersection-over-union (IoU) thresholds. It’s a standard object detection metric.
* Frame Per Second (FPS):
    * Real-time performance requires high FPS. The model should ideally operate at 30 FPS or higher.
* Inference Latency: 
    * The time taken by the model to process a single frame. This should be minimal (e.g., under 30ms) to ensure real-time detection.
* False Positive and False Negative Rates: 
    * Weed detection systems need low false positives (wrong weed detections) and low false negatives (missed weeds).
* IoU (Intersection over Union): 
    * Measures the overlap between the predicted bounding box and the ground truth. Higher IoU indicates better localization accuracy.
* Robustness Evaluation: 
    * Measure performance under varied lighting conditions, angles, motion, and occlusions by testing the model on diverse datasets.

## 2

### 2a

* Field Variability
    * Geographical Diversity: Collect data from different regions with varying soil types, topographies, and ecosystems to account for different species of weeds and crops.
* Weather Conditions
    * Capture images during different weather conditions (sunny, cloudy, rainy, foggy) to account for environmental factors affecting visibility and appearance of crops and weeds.
* Time of Day
    * Take images at different times (morning, afternoon, evening) to capture variations in lighting and shadow effects.
* Growth Stages
    * Collect images of both weeds and crops at various stages of growth (from seedlings to fully grown) to improve model generalization.
    * Include different weed densities and overlaps to simulate real-world challenges where crops and weeds might be intertwined.
* Hardware and Sensor Variation:
    * Use drones with different types of cameras (RGB, multispectral, hyperspectral) and varying resolutions to ensure the model generalizes across devices.
    * Introduce variance in drone altitude and angles to simulate different image capture scenarios.
* Manual Labelling:
    * Utilize human experts (farmers, agronomists) to label weed species in the collected images accurately. These labels should indicate the bounding box, species, and growth stage.
* Synthetic Data Generation:
    * To augment the dataset, use tools like GANs (Generative Adversarial Networks) to generate synthetic images of weeds in crops. This can help fill gaps in the dataset, especially for rare weeds and conditions that are hard to capture.

### 2b

* Data Cleaning:
    * Remove Noise: Eliminate blurry or low-quality images that can degrade model performance.
* Crop and Weed Segmentation: 
    * Use image annotation tools to separate the crops and weeds, ensuring that the labels and bounding boxes are clean and accurate.
* Resizing:
    * Resize all images to a consistent size, depending on the input size of the deep learning model (e.g., 512x512 pixels). This ensures uniformity and reduces computational load.
* Normalization:
    * Normalize pixel values by scaling them to a range of 0-1 or mean-centering them based on the dataset’s mean and standard deviation. This accelerates convergence during model training.
* Data Augmentation:
    * Apply random augmentations during training to simulate real-world variations, including:
        * Rotation and Flipping: To account for different drone angles.
        * Colour Jitter: To adjust brightness, contrast, and saturation, simulating changes in lighting.
        * Random Cropping and Scaling: To simulate various distances between the drone and the crops/weeds.
        * Gaussian Noise: To simulate image sensor noise from the drone camera.
        * Motion Blur:  To mimic drone motion during image capture.
* Label Encoding:
    * For multi-class classification, encode weed species labels as numerical values using one-hot encoding or label encoding for the model to interpret them correctly.
* Data Splitting:
    * Split the dataset into training, validation, and test sets to ensure proper evaluation of the model. A common split would be 70% training, 15% validation, and 15% testing.

### 2c

Class imbalance, where certain weed species are underrepresented, can negatively impact model performance. To mitigate this:

* Oversampling:
    * Use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples of rare weed species, thus balancing the dataset.
* Class Weighting:
    * Assign higher weights to the loss function for underrepresented classes (rare weed species). This forces the model to pay more attention to rare weeds during training.
* Data Augmentation for Rare Classes:
    * Apply more aggressive data augmentation (rotations, flips, color changes) to the rare weed species. This can help generate more training examples for underrepresented classes.
* Focal Loss:
    * Implement focal loss to reduce the weight assigned to well-classified examples and focus on hard-to-classify (rare) examples.
* Resampling the Dataset:
    * Perform down sampling on the majority class (crops or common weeds) or up sampling on the minority class (rare weeds) to reduce imbalance.

### 2d

Automation can streamline the data collection, pre-processing, and model training pipeline in several ways:

* Automated Data Collection:    
    * Autonomous Drones: Use pre-programmed drones equipped with GPS and imaging tools to autonomously collect images over large fields without manual intervention.
    * Edge Processing: Equip drones with basic edge processing capabilities to automatically filter out poor-quality images before storage, reducing the dataset size for downstream processing.
* Automated Labelling:
    * Utilize semi-supervised learning or AI-assisted labelling tools to help human annotators label the weeds faster and with more accuracy. These tools can suggest labels based on pre-trained models.
* Pipeline Automation:
    * Use tools like Apache Airflow or Kubeflow to automate the entire machine learning pipeline, including data ingestion, pre-processing, model training, and deployment. These tools can manage data flows, handle version control, and monitor performance metrics automatically.
* Augmentation Automation:
    * Use libraries like Albumentations or imgaug to automate data augmentation, with predefined strategies for common tasks like rotations, colour jittering, and noise addition.
* Model Retraining Automation:
    * Implement continuous learning pipelines where new drone images are fed back into the system for retraining, ensuring the model stays up-to-date with the latest conditions and weed species.

## 3

### 3a

Multi-spectral imagery is widely used for assessing crop health because it can capture data in several bands (e.g., red, green, blue, near-infrared) beyond what the human eye can see. One of the most effective methods for estimating crop health from multi-spectral imagery is using vegetation indices, particularly the Normalized Difference Vegetation Index (NDVI).

* Capture Multi-Spectral Data:
    * Use a drone or satellite equipped with multi-spectral sensors that can capture light in the red and near-infrared (NIR) bands. Healthy crops typically reflect more NIR light and absorb more red light.
* Calculate NDVI:
    * Landsat Normalized Difference Vegetation Index (NDVI) is used to quantify vegetation greenness and is useful in understanding vegetation density and assessing changes in plant health. 
    * NDVI is calculated using the following formula:
    
        * ''' NDVI = (NIR - Red) / (NIR + Red) '''
            * NIR – light reflected in the near-infrared spectrum
            * RED – light reflected in the red range of the spectrum

    * NDVI values range from -1 to +1. 
    * Healthy vegetation typically produces higher NDVI values (closer to +1), 
    * while stressed or unhealthy crops have lower NDVI values (closer to 0).

* Generate Health Maps:
    * Use the NDVI data to create a heatmap of crop health, where greener areas represent healthy crops, and yellow or brown areas indicate stressed or unhealthy regions.
    * For further granularity, threshold the NDVI values to classify the crops into categories (e.g., healthy, moderate stress, severe stress).
* Use Additional Vegetation Indices:
In addition to NDVI, you can use other indices to capture specific crop health indicators:
    * Enhanced Vegetation Index (EVI): Improves sensitivity in high biomass areas.
    * Normalized Difference Red Edge (NDRE): Effective for identifying early signs of stress before they are visible to the naked eye.
    * Soil-Adjusted Vegetation Index (SAVI): Reduces soil influence in areas with sparse vegetation.
* Temporal Analysis:
    * Compare NDVI values over time (weekly, monthly) to monitor the crop’s growth and health trajectory, identifying patterns of decline or improvement.
* Ground Truthing:
    * Validate the NDVI-based crop health estimates with actual ground truth data, such as leaf health measurements, crop yields, or soil moisture levels.

### 3b

To optimize crop management and prioritize treatment areas, integrating crop health data with weed detection data can be highly effective. Here’s a method to achieve this integration:

* Create Combined Maps:
    * Combine the weed detection output (bounding boxes, labels) with the crop health heatmap (NDVI or similar indices) to visualize both weed locations and crop health in the same field map.
    * Use colour coding to distinguish between areas with healthy crops, stressed crops, and weed-infested areas:
        * Green for healthy crops
        * Yellow/Red for stressed or unhealthy crops
        * Orange/Red boxes for weed detections

* Assign Priority Scores:
    * Assign a priority score to different sections of the field based on the intersection of weed density and crop health:
        * High-priority areas: Sections where weeds are detected, and crop health is poor (low NDVI values). These areas are likely to require immediate treatment (herbicides, irrigation, or nutrient adjustments).
        * Moderate-priority areas: Regions where crop health is poor, but no significant weed infestation is detected. These areas may need attention to improve soil conditions or address pest-related stress.
        * Low-priority areas: Sections where crops are healthy, regardless of the presence of weeds. Weed treatment might be delayed here if crop health is unaffected.

* Real-Time Alerts:
    * Set up an automated alert system that notifies farmers or agronomists when high-priority areas are detected. This can be integrated into a farm management system that provides suggestions for treatments (e.g., herbicide application or irrigation) based on both weed density and crop stress.

* Dynamic Treatment Zones:
    * Based on the combined weed and crop health maps, define treatment zones dynamically. These zones can be used for precision agriculture where the drone or machinery applies treatments only to areas that need it, reducing over-application of herbicides and improving efficiency.
        * Example: A field sprayer can use GPS-based control to apply herbicides only in areas marked as high priority (weed + stressed crops).

* Temporal Integration:
    * By comparing crop health data over time and overlaying weed detection results from different time periods, you can identify patterns of weed infestation and crop health decline. This helps in predicting future areas of concern and optimizing treatment schedules.

* Use AI for Prioritization:
    * Build a decision-making model (e.g., a rules-based system or reinforcement learning) that automatically prioritizes areas based on weed density, crop stress, and economic factors (cost of treatment vs potential yield loss). This model can recommend whether to focus on weed removal, crop health improvement, or both.

* Farm Dashboard Integration:
    * Present this integrated data via a farm dashboard where farmers can visualize the field in real time, monitor treatment suggestions, and manually override the system’s recommendations if needed. This allows for flexibility in decision-making.

## 5

### 5a 

* Visual Explanations with Saliency Maps:
    * Use Grad-CAM (Gradient-weighted Class Activation Mapping) or saliency maps to highlight the specific regions in the input image that contributed the most to the model’s decision. This would help farmers see which parts of the image the model identifies as weeds or unhealthy crops.

* Bounding Box and Label Visualization:
    * Display the detected weeds and crops using bounding boxes in real-time, alongside species labels and health status. Farmers can easily see which areas the model identifies as containing weeds and understand what is being detected.

* Decision Trees or Rule-based Explanations:
    * Simplify the decision-making process by translating complex model decisions into understandable rules. For example, provide explanations like: “The model identified this region as weeds because it matches the colour and texture patterns typical of weed species X.”

* Confidence Scores:
    * Show confidence scores for each detection or classification to give an indication of how certain the model is in its decision. This allows farmers to decide whether further manual inspection is needed.

* Comparison with Ground Truth Data:
    * Provide comparative analysis against known labels or farmer input to show how closely the model aligns with human assessment. This can help build trust, especially in cases where the model’s predictions match real-world observations.

* Model Simplification with SHAP or LIME:
    * Use SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to break down how different features (e.g., colour, shape, texture) influenced the model’s predictions. This method can help explain individual predictions in a human-friendly manner.

* Health and Growth Reports:
    * For crop health models, generate easy-to-understand reports indicating the overall health of the crops (e.g., “75% of the crop area is healthy, while 10% shows signs of stress”). These reports can be customized for specific growth stages or environmental factors.

* Overlaying Information on Field Maps:
    * Integrate weed detection and crop health analysis into geo-referenced maps that agronomists and farmers use. By visualizing the weed/crop health data directly on these maps, users can see problem areas in the field and take action accordingly.

* Actionable Insights:
    * Offer actionable suggestions based on the model’s outputs, such as “weed infestation detected in 10% of the field—consider herbicide application” or “crop stress detected, adjust irrigation levels.” This makes the model’s output directly useful to decision-making.

## 5b

* Grad-CAM (Gradient-weighted Class Activation Mapping):
    * Grad-CAM is a powerful tool to visualize which parts of the image are most relevant to the model’s prediction. 
    * For a weed detection task:
        * Overlay a heatmap on the input image, where hotter regions indicate areas that contributed most to the model’s classification of weeds.
        * This shows the farmer precisely which parts of the image were used to identify a weed.

* Saliency Maps:
    * Saliency maps highlight pixels that the model considers important for making predictions. For a given input image:
    * Generate a grayscale saliency map where the intensity of each pixel indicates its influence on the model’s decision.
    * This helps explain why the model identified a particular region as containing weeds or stressed crops.

* Bounding Box and Label Overlay:
    * Show the weed detection results by drawing bounding boxes around detected weeds and labeling them with the weed species or classification confidence.
    * This makes it clear to farmers which areas of the field the model is focusing on.
    * For crop health detection, you can visualize areas of healthy versus stressed crops.

* Feature Visualization:
    * For each detected object (weed or crop), visualize the key features (e.g., texture, color) that the model used for detection. This can be done by identifying feature maps at different layers of the CNN, showing which patterns or textures contributed to the decision.

* Side-by-Side Comparison:
    * Display the input image alongside the model’s output (weed detection, health assessment) in a before-and-after style, showing the original image and the model’s interpretation on the same screen.

* Region of Interest (ROI) Zoom:
    * For images where weeds are detected in complex or dense areas, provide a zoomed-in view of specific regions of interest (ROI) where the model detected weeds, along with explanations (e.g., “weed detected due to irregular leaf shape and color”).

* Layer-wise Visualization:
    * In cases where deeper understanding is required, you can visualize intermediate layers of the neural network to show how the model extracts features at different levels of abstraction. This would be more relevant for technical users like agronomists.

* Video Playback with Annotations:
    * For drone footage, provide real-time video playback where the model’s decisions are shown frame by frame with bounding boxes a