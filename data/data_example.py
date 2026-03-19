import numpy as np
import pickle
import os

abnormality_category = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum","Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax"]


disease_prediction_category = ["Mortality", "Sepsis", "Respiratory Failure", "Heart Failure"]

# Get absolute path to the example image to simulate real data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_IMAGE_PATH = os.path.join(current_dir, "example.jpg")

def generate_patient_train_data(patient_idx, mode="train"):
    # Training & Validation & Testing Data
    # 模式：train/validate/test
    # 结构：单张 reference image -> 单张 target image (预测) + Lab test ([ref_time, target_time])
    
    if mode == "train":
        suffix = "train"
    elif mode == "validate":
        suffix = "val"
    elif mode == "test":
        suffix = "test"
    else:
        suffix = "val"
        
    patient_id = f"patient_{patient_idx}_{suffix}"
    
    # Reference Image
    reference_image_id = f"{patient_id}_ref_img"
    # 使用真实存在的 example.jpg 模拟
    reference_image_path = EXAMPLE_IMAGE_PATH
    
    # reference_image_time 在 [0, 99) 之间
    reference_image_time = np.float32(np.random.randint(0, 100))
    reference_image_abnormality = np.random.randint(0, 2, size=len(abnormality_category)).astype(np.float32)

    # Target Image
    target_image_id = f"{patient_id}_target_img"
    # 使用真实存在的 example.jpg 模拟
    target_image_path = EXAMPLE_IMAGE_PATH
    
    max_time_horizon = 100
    if reference_image_time >= max_time_horizon - 1:
        target_image_time = reference_image_time
    else:
        target_image_time = np.float32(np.random.randint(reference_image_time, max_time_horizon))
        
    target_image_abnormality = np.random.randint(0, 2, size=len(abnormality_category)).astype(np.float32)

    # Lab Test Data
    # [reference_image_time, target_image_time] 闭区间
    if target_image_time < reference_image_time:
        target_image_time = reference_image_time
        
    lab_test_time = np.arange(reference_image_time, target_image_time + 1, 1).astype(np.float32)
    
    # 53 features
    num_features = 53
    lab_test_data = np.random.rand(len(lab_test_time), num_features).astype(np.float32)
    
    missingness_prob = 0.2
    lab_test_mask = (np.random.rand(len(lab_test_time), num_features) > missingness_prob).astype(np.float32)

    patient_data = {
        "patient_id": patient_id,
        "reference_image_id": reference_image_id,
        "reference_image_path": reference_image_path,
        # "reference_image": reference_image, # Removed as per instruction
        "reference_image_time": reference_image_time,
        "reference_image_abnormality": reference_image_abnormality,
        
        "target_image_id": target_image_id,
        "target_image_path": target_image_path,
        # "target_image": target_image, # Removed as per instruction
        "target_image_time": target_image_time,
        "target_image_abnormality": target_image_abnormality,
        
        "lab_test_time": lab_test_time,
        "lab_test_data": lab_test_data,
        "lab_test_mask": lab_test_mask,
    }
    return patient_data

def generate_patient_test_data(patient_idx):
    # Testing Data Generation
    patient_id = f"patient_{patient_idx}_test"
    
    # Reference Image
    reference_image_id = f"{patient_id}_ref_img"
    reference_image_path = EXAMPLE_IMAGE_PATH
    
    # Test时只取前48小时的数据
    reference_image_time = np.float32(int(np.random.randint(0, 48)))
    reference_image_abnormality = np.random.randint(0, 2, size=len(abnormality_category)).astype(np.float32)

    # Target Times
    max_targets = 5
    ref_t_int = int(reference_image_time)
    all_times = np.arange(0, 48, 1, dtype=np.int32)
    candidate_target_times = all_times[all_times != ref_t_int]
    
    num_targets = int(np.random.randint(1, min(max_targets, len(candidate_target_times)) + 1))
    
    target_image_time_list = np.sort(
        np.random.choice(candidate_target_times, size=num_targets, replace=False)
    ).astype(np.float32)
    

    # Target Image Paths (Simulated)
    target_image_path_list = [EXAMPLE_IMAGE_PATH for _ in range(num_targets)]
    
    # Lab Test Data (Full 0-48h)
    lab_test_time = np.arange(0, 48, 1, dtype=np.int32).astype(np.float32)
    num_features = 53
    lab_test_data = np.random.rand(len(lab_test_time), num_features).astype(np.float32)
    missingness_prob = 0.2
    lab_test_mask = (np.random.rand(len(lab_test_time), num_features) > missingness_prob).astype(np.float32)

    # Disease Prediction Label
    disease_prediction_label = np.random.randint(0, 2, size=len(disease_prediction_category)).astype(np.float32)

    # Common data
    base_data = {
        "patient_id": patient_id,
        "reference_image_id": reference_image_id,
        "reference_image_path": reference_image_path,
        "reference_image_time": reference_image_time,
        "reference_image_abnormality": reference_image_abnormality,
        
        "lab_test_time": lab_test_time,
        "lab_test_data": lab_test_data,
        "lab_test_mask": lab_test_mask,
        
        "disease_prediction_label": disease_prediction_label
    }

    # Version 0: Raw Data (Before target time generation)
    # 不包含 target times, 也不包含 generated images
    # 仅包含 reference image 和 lab test info
    v0_raw_data = base_data.copy()

    # Version 1: Input for Generator (Pre-generation)
    # 包含 target times, 但没有 generated images (path)
    # 可以包含 Ground Truth paths 如果是为了 evaluation, 但这里模拟 "Before Generation" 状态
    v1_input_data = base_data.copy()
    v1_input_data.update({
        "target_image_time_list": target_image_time_list,
        # "target_image_path_list": ... # Not present yet or None
    })

    # Version 2: Output of Generator (Post-generation)
    # 包含 generated images (paths)
    v2_output_data = base_data.copy()
    v2_output_data.update({
        "target_image_time_list": target_image_time_list,
        "target_image_path_list": target_image_path_list, # Points to generated images

    })
    
    return v0_raw_data, v1_input_data, v2_output_data

def main():
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)

    # Note: Using Dictionary structure {patient_id: data_dict} to match typical Dataloader expectations
    
    # Generate Train Data
    print("Generating training data...")
    train_data = {}
    for i in range(10):
        p_data = generate_patient_train_data(i, mode="train")
        train_data[p_data['patient_id']] = p_data
    
    train_pkl_path = os.path.join(save_dir, "train_data_for_gen_develop.pkl")
    with open(train_pkl_path, "wb") as f:
        pickle.dump(train_data, f)
    print(f"Saved training data to {train_pkl_path}")

    # Generate Validation Data
    print("Generating validation data...")
    val_data = {}
    for i in range(5):
        p_data = generate_patient_train_data(i, mode="validate")
        val_data[p_data['patient_id']] = p_data
        
    val_pkl_path = os.path.join(save_dir, "val_data_for_gen_develop.pkl")
    with open(val_pkl_path, "wb") as f:
        pickle.dump(val_data, f)
    print(f"Saved validation data to {val_pkl_path}")

    # Generate Test Data (For Generator Training)
    # 目的：用于 Generator 训练过程中的 Evaluation (与 Train/Val 格式一致)
    # 这部分数据用于在训练 Generator 时，计算 Test Set 上的指标 (如 L1, FID 等)
    print("Generating test data for generator training...")
    test_gen_data = {}
    for i in range(5):
        p_data = generate_patient_train_data(i, mode="test")
        test_gen_data[p_data['patient_id']] = p_data

    test_gen_pkl_path = os.path.join(save_dir, "test_data_for_gen_develop.pkl")
    with open(test_gen_pkl_path, "wb") as f:
        pickle.dump(test_gen_data, f)
    print(f"Saved test data for generator training to {test_gen_pkl_path}")

    # Generate Inference Simulation Data (Three Versions)
    # 目的：模拟完整的 Inference Pipeline (Patient -> Time Pred -> Generator -> Disease Pred)
    # 之前的 "Test Data" 主要用于这个目的
    print("Generating inference simulation data...")
    test_raw_data = {}   # Version 0
    test_input_data = {} # Version 1
    test_output_data = {} # Version 2
    
    for i in range(5):
        v0, v1, v2 = generate_patient_test_data(i)
        test_raw_data[v0['patient_id']] = v0
        test_input_data[v1['patient_id']] = v1
        test_output_data[v2['patient_id']] = v2
    
    test_v0_path = os.path.join(save_dir, "data_for_gen_infer.pkl")
    with open(test_v0_path, "wb") as f:
        pickle.dump(test_raw_data, f)
    print(f"Saved inference test data (Version 0 - Raw) to {test_v0_path}")
        
    test_v1_path = os.path.join(save_dir, "data_for_gen_infer_with_tar_points.pkl")
    with open(test_v1_path, "wb") as f:
        pickle.dump(test_input_data, f)
    print(f"Saved inference test data (Version 1 - Input) to {test_v1_path}")
    
    test_v2_path = os.path.join(save_dir, "data_for_for_gen_infer_with_tar_img.pkl")
    with open(test_v2_path, "wb") as f:
        pickle.dump(test_output_data, f)
    print(f"Saved inference test data (Version 2 - Output) to {test_v2_path}")

if __name__ == "__main__":
    main()


# 数据文件说明：
# 
# A. Generator Training/Development Sets (用于 Generator 的模型训练与评估)
#    这些数据都采用 Pair-wise 结构 (Ref Image + Lab Sequence -> Target Image)，用于监督学习。
#    - train_data_for_generator_develop.pkl: 训练集
#    - val_data_for_generator_develop.pkl: 验证集
#    - test_data_for_generator_develop.pkl: 测试集 (新增，用于计算 FID, L1 等生成指标)
#
# B. Inference Pipeline Simulation Sets (用于模拟实际推理全流程)
#    这组数据模拟了 "Patient -> Time Prediction -> Image Generation -> Disease Prediction" 的流程。
#    - test_data.pkl: 原始病人数据 (Lab + Ref Image)，不包含预测目标时间点。
#    - test_data_with_target_points.pkl: (模拟) 经过 Time Predictor 后，确定了需要生成的时刻 (Target Times)。这是 Generator 的 Inference 输入。
#    - test_data_for_disease_pred.pkl: (模拟) 经过 Generator 后，生成的图像数据。这是 Disease Predictor 的输入。
#
# 总结：
# 如果你只关心 Generator 的训练，请使用 A 类数据 (Train/Val/Test)。
# 如果你要测试整个系统的串联 (Inference)，请关注 B 类数据。