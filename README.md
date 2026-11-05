# visionartificial

# Computer Vision: Age Estimation for Retail Compliance 📊

### 📌 Project Overview
This project develops a Deep Learning regression model using computer vision to estimate customer age from facial images. Designed for supermarket self-checkout environments, the system serves as an operational screening tool to optimize identity verification workflows for age-restricted sales (alcohol and tobacco).

### 🛠️ Technical Stack & Frameworks
* **Core Infrastructure:** Python, TensorFlow, Keras, Jupyter Notebook.
* **Architecture:** ResNet50 pre-trained on ImageNet (Transfer Learning & Fine-Tuning).
* **Data Pipeline:** Refactored continuous flow optimization using `flow_from_dataframe` with exact continuous targets (`class_mode='raw'`).
* **Regularization & Graphics:** Pixel normalization (1./255), horizontal flip data augmentation, Matplotlib, and Seaborn for Exploratory Data Analysis (EDA).

### 📊 Experimental Setup & Dataset Volume
* **Total Dataset:** 7,591 images.
* **Data Split (25% Validation Split):**
  * **Training Subset:** 5,694 images (augmented with horizontal flips to prevent overfitting).
  * **Validation Subset:** 1,897 images (isolated from augmentation to maintain testing integrity and prevent data leakage).
* **Training Execution:** 5 epochs utilizing the Adam optimizer with a controlled learning rate ($0.0001$) for steady weights fine-tuning.

### 🏆 Key Metrics & Validation Results
* **Final Validation MAE:** **5.9241** (Mean Absolute Error).
* **Target Achievement:** Significantly outperformed the mandatory project threshold (MAE ≤ 8). On average, the model's age prediction deviates by less than 6 years.

### ⚠️ Critical Business Limitations & Operational Guardrails
* **Insufficiency of Aggregate Metrics:** A global MAE of 5.92 years represents a macro-average and cannot guarantee uniform precision near the critical legal threshold of 18 years. Due to dataset distribution, residual errors near the minority-age boundary could lead to false negatives.
* **Operational Deployment:** The system is explicitly positioned as a supportive screening tool (e.g., triggering a mandatory human ID check only if the estimated age is under 25) and under no circumstances replaces definitive human physical verification for legal compliance.
* **Future Commercial Expansion:** Applications regarding behavioral demographics or data-driven marketing profiles remain as future development pipelines, requiring independent sub-group metric validation before full commercial deployment.

### 📂 Repository Structure
* `notebook.ipynb` - Complete Jupyter Notebook containing the data exploration, pipeline refactoring, and model architecture.
* `run_model_on_gpu.py` - Dynamically compiled standalone Python script optimized to execute the verified functions on remote GPU platforms.
