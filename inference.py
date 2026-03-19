import argparse
import os
import yaml
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from datasets.dataloader import InferenceDataset, CollateFunc
from train import ControlGenModel
from diffusers import DDPMScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Generator Inference Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file (e.g., configs/stage2_config.yaml)")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.ckpt)")
    parser.add_argument('--input_data', type=str, required=True, help="Path to input data pickle (with target points)")
    parser.add_argument('--output_dir', type=str, default="results/inference", help="Directory to save generated images")
    parser.add_argument('--output_pkl', type=str, default="data/data_for_gen_infer_with_tar_img.pkl", help="Path to save the final output pickle")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use (cuda or cpu)")
    return parser.parse_args()

def load_model(config_path, checkpoint_path, device):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure stage is 2 for inference
    config['stage'] = 2
    
    print(f"Loading model from {checkpoint_path}")
    # Load model from checkpoint
    model = ControlGenModel.load_from_checkpoint(checkpoint_path, config=config)
    model.to(device)
    model.eval()
    
    return model, config

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    model, config = load_model(args.config, args.checkpoint, device)
    num_inference_steps = int(config.get('num_inference_steps', 50))
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be a positive integer.")
    
    # 2. Load Dataset
    print(f"Loading inference data from {args.input_data}")
    dataset = InferenceDataset(
        data_pkl_path=args.input_data,
        resize=config.get('resize', 512),
        crop=config.get('crop', 512)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1, # Process one patient at a time for simplicity with variable target times
        num_workers=config.get('num_workers', 4),
        shuffle=False,
        collate_fn=CollateFunc()
    )
    
    # 3. Prepare Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dictionary to store results for final pickle
    results_dict = {}
    # We need to load the original data to preserve other fields or just update what we have?
    # The InferenceDataset loads the pickle. We can just create a new dict with updated fields.
    # But usually we want to keep the original structure. 
    # Let's load the original data dict to update it.
    with open(args.input_data, 'rb') as f:
        original_data = pickle.load(f)
    
    print("Starting inference...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Unpack batch (batch_size=1)
            patient_id = batch['patient_id'][0] # String
            lab_test_full = batch['lab_test'].to(device) # (1, L_full, D)
            lab_times_full = batch['lab_times'].to(device) # (1, L_full)
            ref_img = batch['ref_img'].to(device) # (1, C, H, W)
            ref_time = batch['ref_time'].to(device) # (1,)
            adapter_residuals = model._get_ref_adapter_residuals(ref_img)
            
            # Get target times
            if 'target_times' not in batch or batch['target_times'][0] is None:
                print(f"Skipping {patient_id}: No target times found.")
                continue
            
            target_times = batch['target_times'][0] # List of floats or Tensor
            if isinstance(target_times, torch.Tensor):
                target_times = target_times.tolist()
            
            generated_paths = []
            
            # Iterate over each target time
            for t_idx, target_time in enumerate(target_times):
                target_time = float(target_time)
                
                # 1. Slice lab data between ref_time and target_time.
                # For forward generation (target >= ref), we keep chronological order.
                # For backward generation (target < ref), we reverse the selected interval
                # so the sequence still starts at ref_time and ends at target_time.
                if target_time >= float(ref_time.item()):
                    interval_start, interval_end = float(ref_time.item()), target_time
                    reverse_interval = False
                else:
                    interval_start, interval_end = target_time, float(ref_time.item())
                    reverse_interval = True

                # Create mask for time slicing
                # lab_times_full is (1, L)
                time_mask = (lab_times_full >= interval_start) & (lab_times_full <= interval_end)
                
                # Check if we have any data in the selected interval.
                # If not, fall back to the lab observation exactly at ref_time so inference can proceed.
                if time_mask.sum() == 0:
                     time_mask = (lab_times_full == ref_time)
                
                # Slice the tensors
                # Since batch_size=1, we can just mask the dimension 1
                valid_indices = torch.nonzero(time_mask[0], as_tuple=True)[0]
                
                if len(valid_indices) == 0:
                     print(f"Warning: No valid lab data for {patient_id} at t={target_time}. Skipping.")
                     generated_paths.append(None)
                     continue

                if reverse_interval:
                    valid_indices = torch.flip(valid_indices, dims=[0])
                
                # Slice and keep batch dim
                lab_test = lab_test_full[:, valid_indices, :]
                # Update length to match slice
                lab_lengths = torch.tensor([len(valid_indices)]).long().to(device)
                seq_mask = model._build_sequence_mask(lab_lengths, lab_test.shape[1], device)
                
                # Target index is the last element of the sliced sequence
                t_gen = (lab_lengths - 1).view(1, 1)

                # --- Inference Steps ---
                
                # 1. Encode Lab Tests
                lab_features = model.lab_test_encoder.encode_per_lab_test(lab_test, padding_masks=seq_mask)
                seq_length = lab_features.shape[1]
                
                # 2. Knowledge Transform (Lab -> Organ)
                organ_states_from_lab, organ_mask_from_lab = model.knowledge_transform.lab_to_organs(
                    lab_features,
                    model.lab_test_names,
                    time_mask=seq_mask
                )
                
                # 3. Encode Ref Image -> Initial Organ State
                abn_features_0 = model.cxr_encoder(ref_img) # (1, num_abnormalities, abn_feat_dim)
                
                # Expand abnormality features to match time sequence
                abn_features_seq = torch.zeros(
                    1, seq_length, len(model.abnormality_names), model.abn_feat_dim,
                    device=device, dtype=abn_features_0.dtype
                )
                abn_features_seq[:, 0] = abn_features_0
                
                # 4. Knowledge Transform (Abn -> Organ)
                abn_time_mask = torch.zeros(1, seq_length, dtype=torch.bool, device=device)
                abn_time_mask[:, 0] = True
                
                organ_states_from_abn, organ_mask_from_abn = model.knowledge_transform.abn_to_organs(
                    abn_features_seq,
                    model.abnormality_names,
                    time_mask=abn_time_mask
                )
                
                # Combine Organ States
                organ_mask = organ_mask_from_lab | organ_mask_from_abn
                organ_states = organ_states_from_lab.clone()
                organ_states[organ_mask_from_abn] = organ_states_from_abn[organ_mask_from_abn]
                
                # 5. Trajectory Encoder
                encoded_organ_states = model.trajectory_encoder(organ_states, organ_mask)
                
                # 6. Infer at target time
                inferred_organ_states = model.trajectory_encoder.infer_at_time(
                    encoded_organ_states,
                    organ_mask,
                    t_gen
                ).squeeze(1) # (1, num_organs, feat_dim)
                
                # 7. Knowledge Transform (Organ -> Abn)
                gen_mask = organ_mask[0, t_gen.squeeze(1)] # (1, num_organs)
                
                inferred_abn_features = model.knowledge_transform.organs_to_abnormalities(
                    inferred_organ_states,
                    gen_mask,
                    model.abnormality_names
                )
                
                # 8. Project to UNet Conditioning
                abn_features_flat = inferred_abn_features.view(1, -1)
                unet_conditioning = model.abn_to_unet_proj(abn_features_flat)
                unet_conditioning = unet_conditioning.view(1, 77, 768)
                
                # 9. Diffusion Generation
                # Initialize random noise
                vae_scale_factor = 2 ** (len(model.vae.config.block_out_channels) - 1)
                latent_shape = (
                    1,
                    model.unet.config.in_channels,
                    ref_img.shape[-2] // vae_scale_factor,
                    ref_img.shape[-1] // vae_scale_factor,
                )
                latents = torch.randn(latent_shape, device=device, dtype=unet_conditioning.dtype)
                
                # Set scheduler timesteps
                model.scheduler.set_timesteps(num_inference_steps, device=device)
                
                # Diffusion Loop
                for t in model.scheduler.timesteps:
                    latent_model_input = model.scheduler.scale_model_input(latents, t)
                    unet_kwargs = {"encoder_hidden_states": unet_conditioning}
                    if adapter_residuals is not None:
                        unet_kwargs["down_intrablock_additional_residuals"] = [state.clone() for state in adapter_residuals]
                    
                    # Predict noise
                    noise_pred = model.unet(latent_model_input, t, **unet_kwargs).sample
                    
                    # Step
                    latents = model.scheduler.step(noise_pred, t, latents).prev_sample
                
                # 10. Decode Latents -> Image
                generated_latents = 1 / model.vae.config.scaling_factor * latents
                generated_image = model.vae.decode(generated_latents).sample
                
                # Clamp and convert to image
                generated_image = (generated_image / 2 + 0.5).clamp(0, 1)
                generated_image = generated_image.cpu().permute(0, 2, 3, 1).float().numpy() # (1, H, W, C)
                img_array = (generated_image[0] * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                
                # 11. Save Image
                save_name = f"{patient_id}_t{int(target_time)}.png"
                save_path = os.path.join(args.output_dir, save_name)
                img.save(save_path)
                
                generated_paths.append(save_path)
            
            # Store results for this patient
            results_dict[patient_id] = generated_paths

    # Save Final Pickle
    print(f"Saving final results to {args.output_pkl}")
    # Update original data with paths
    output_data_final = original_data
    for pid, paths in results_dict.items():
        if pid in output_data_final:
            output_data_final[pid]['target_image_path_list'] = paths
            
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(output_data_final, f)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
