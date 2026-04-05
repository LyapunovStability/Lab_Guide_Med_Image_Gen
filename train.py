import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.dataloader import GeneratorTrainDataset, CollateFunc

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from models.Lab_Test_Encoder import Lab_Test_Encoder
from models.CXR_Encoder import CXR_Encoder
from models.OrganGraph import OrganGraph
from models.KnowledgeGuidedTransform import KnowledgeGuidedTransform
from models.ref_image_adapter import RefImageAdapter
from models.TrajectoryEncoder import TrajectoryEncoder
from models.TimePointSelector import TimePointSelector
from torchvision.utils import save_image
from utils.project_paths import resolve_pubmedbert_source, resolve_sd_model_source

class ControlGenDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _image_root_path(self):
        return self.config.get('image_root_path') or self.config.get('image_root_dir')

    def setup(self, stage=None):
        # [REFRACTOR] Use GeneratorTrainDataset for Stage 1 & 2 Training (Pair-wise)
        # Note: InferenceDataset integration is deferred.
        root = self._image_root_path()
        
        # Generator Training - Train Set
        self.train_dataset = GeneratorTrainDataset(
            data_pkl_path=self.config['train_lab_test_pkl_path'],
            resize=self.config.get('resize', 512),
            crop=self.config.get('crop', 512),
            image_root_path=root,
        )
        # Generator Training - Validation Set
        self.val_dataset = GeneratorTrainDataset(
            data_pkl_path=self.config['val_lab_test_pkl_path'],
            resize=self.config.get('resize', 512),
            crop=self.config.get('crop', 512),
            image_root_path=root,
        )
        
        # Generator Training - Test Set (For Generator Evaluation like FID, L1)
        # Note: This is DIFFERENT from the Inference Pipeline test set.
        self.test_dataset = GeneratorTrainDataset(
            data_pkl_path=self.config['test_lab_test_pkl_path'],
            resize=self.config.get('resize', 512),
            crop=self.config.get('crop', 512),
            image_root_path=root,
        )

        # [TODO] Generator Inference (Inference Pipeline)
        # self.inference_dataset = InferenceDataset(...)
        self.inference_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=True,
            collate_fn=CollateFunc(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=CollateFunc(),
        )

    def test_dataloader(self):
        # Return Test Set DataLoader for Generator Evaluation
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=CollateFunc(),
        )

    # [TODO] Add inference_dataloader() later for Generator Inference
    # def inference_dataloader(self):
    #     ...

class ControlGenModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        for lr_key in ("learning_rate", "adapter_learning_rate", "unet_learning_rate"):
            if lr_key in self.config and self.config[lr_key] is not None:
                self.config[lr_key] = float(self.config[lr_key])
        self.stage = config['stage']
        
        # Feature dimensions
        self.lab_feat_dim = config.get('lab_feat_dim', 512)
        self.abn_feat_dim = config.get('abn_feat_dim', 512)
        self.organ_feat_dim = config.get('organ_feat_dim', 512)
        self.num_abnormalities = config.get('num_abnormalities', 12)
        
        # Load organ graph
        graph_path = config.get('graph_path', 'graph/organ_graph.json')
        self.organ_graph = OrganGraph(graph_path)
        
        # Get lab test names from graph
        self.lab_test_names = self.organ_graph.get_all_lab_tests()
        self.abnormality_names = self.organ_graph.get_all_abnormalities()

        # Knowledge-guided text encoder configuration
        self.knowledge_concept_emb_dim = config.get('knowledge_concept_emb_dim', 768)
        self.knowledge_use_text_encoder = config.get('knowledge_use_text_encoder', True)
        if self.knowledge_use_text_encoder:
            self.knowledge_text_encoder_model_name = resolve_pubmedbert_source(
                config.get('knowledge_text_encoder_model_name')
            )
        else:
            self.knowledge_text_encoder_model_name = config.get('knowledge_text_encoder_model_name')
        self.knowledge_text_max_length = config.get('knowledge_text_max_length', 512)
        self.reuse_knowledge_concept_buffers_from_ckpt = config.get(
            'reuse_knowledge_concept_buffers_from_ckpt', False
        )

        if self.stage == 1:
            # Stage 1: Trajectory Module Pre-training
            self.cxr_encoder = CXR_Encoder(
                task='na',
                pretrained=config.get(
                    'cxr_encoder_pretrained',
                    config.get('pretrained', True),
                ),
                num_abnormalities=self.num_abnormalities,
                feature_dim=self.abn_feat_dim,
                weights_path=config.get(
                    'cxr_encoder_weights_path',
                    config.get('eva_x_weights_path', None),
                ),
            )
            
            self.lab_test_encoder = Lab_Test_Encoder(
                max_len=100,
                d_model=self.lab_feat_dim,
                n_heads=8,
                num_layers=4,
                dim_feedforward=2048,
                feat_dim=len(self.lab_test_names) if self.lab_test_names else 53
            )
            
            # Knowledge-guided transformation module (device will be set after model moves to device)
            self.knowledge_transform = KnowledgeGuidedTransform(
                organ_graph=self.organ_graph,
                lab_feat_dim=self.lab_feat_dim,
                abn_feat_dim=self.abn_feat_dim,
                organ_feat_dim=self.organ_feat_dim,
                concept_emb_dim=self.knowledge_concept_emb_dim,
                use_clip=self.knowledge_use_text_encoder,
                text_encoder_model_name=self.knowledge_text_encoder_model_name,
                text_max_length=self.knowledge_text_max_length,
                device='cpu'  # Will be updated when model moves to device
            )
            
            # Trajectory encoder with GAT
            self.trajectory_encoder = TrajectoryEncoder(
                organ_feat_dim=self.organ_feat_dim,
                num_layers=config.get('trajectory_num_layers', 3),
                gnn_num_layers=config.get('gnn_num_layers', 2),
                num_heads=config.get('num_heads', 4),
                dropout=config.get('dropout', 0.1)
            )
            
            # Time point selection (optional for Stage 1, can be trained separately)
            self.time_point_selector = TimePointSelector(
                hidden_dim=config.get('time_selector_hidden_dim', 64),
                num_ode_layers=config.get('time_selector_num_layers', 3)
            )
            
            # Freeze pretrained ViT only; train abnormality_projection in Stage 1 (see configure_optimizers).
            for param in self.cxr_encoder.vision_backbone.parameters():
                param.requires_grad = False
        
        elif self.stage == 2:
            # Stage 2: Diffusion Model Training with Trajectory Fine-tuning
            sd_root = resolve_sd_model_source(config.get('model_id'))
            self.vae = AutoencoderKL.from_pretrained(sd_root, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(sd_root, subfolder="unet")

            # Initialize CXR Encoder (Frozen)
            self.cxr_encoder = CXR_Encoder(
                task='na',
                pretrained=config.get(
                    'cxr_encoder_pretrained',
                    config.get('pretrained', True),
                ),
                num_abnormalities=self.num_abnormalities,
                feature_dim=self.abn_feat_dim,
                weights_path=config.get(
                    'cxr_encoder_weights_path',
                    config.get('eva_x_weights_path', None),
                ),
            )
            # CXR: load projection (and backbone if present) from Stage 1 when available, then freeze all.
            for param in self.cxr_encoder.parameters():
                param.requires_grad = False
            
            self.lab_test_encoder = Lab_Test_Encoder(
                max_len=100,
                d_model=self.lab_feat_dim,
                n_heads=8,
                num_layers=4,
                dim_feedforward=2048,
                feat_dim=len(self.lab_test_names) if self.lab_test_names else 53
            )
            
            # Knowledge-guided transformation module
            self.knowledge_transform = KnowledgeGuidedTransform(
                organ_graph=self.organ_graph,
                lab_feat_dim=self.lab_feat_dim,
                abn_feat_dim=self.abn_feat_dim,
                organ_feat_dim=self.organ_feat_dim,
                concept_emb_dim=self.knowledge_concept_emb_dim,
                use_clip=self.knowledge_use_text_encoder,
                text_encoder_model_name=self.knowledge_text_encoder_model_name,
                text_max_length=self.knowledge_text_max_length,
                device='cpu'
            )
            
            # Trajectory encoder with GAT
            self.trajectory_encoder = TrajectoryEncoder(
                organ_feat_dim=self.organ_feat_dim,
                num_layers=config.get('trajectory_num_layers', 3),
                gnn_num_layers=config.get('gnn_num_layers', 2),
                num_heads=config.get('num_heads', 4),
                dropout=config.get('dropout', 0.1)
            )
            
            # Time point selection
            self.time_point_selector = TimePointSelector(
                hidden_dim=config.get('time_selector_hidden_dim', 64),
                num_ode_layers=config.get('time_selector_num_layers', 3)
            )
            
            # Load pre-trained trajectory module from Stage 1
            if 'stage1_checkpoint_path' in config and os.path.exists(config['stage1_checkpoint_path']):
                checkpoint = torch.load(config['stage1_checkpoint_path'], map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    # Load lab_test_encoder
                    lab_test_state_dict = {k.replace('lab_test_encoder.', ''): v 
                                          for k, v in state_dict.items() 
                                          if k.startswith('lab_test_encoder.')}
                    self.lab_test_encoder.load_state_dict(lab_test_state_dict, strict=False)
                    # Load knowledge_transform
                    kg_state_dict = {k.replace('knowledge_transform.', ''): v 
                                   for k, v in state_dict.items() 
                                   if k.startswith('knowledge_transform.')}
                    if not self.reuse_knowledge_concept_buffers_from_ckpt:
                        concept_buffer_keys = {
                            'lab_concept_emb',
                            'abn_concept_emb',
                            'org_concept_emb',
                            'relation_emb',
                        }
                        kg_state_dict = {
                            k: v for k, v in kg_state_dict.items() if k not in concept_buffer_keys
                        }
                    self.knowledge_transform.load_state_dict(kg_state_dict, strict=False)
                    # Load trajectory_encoder
                    traj_state_dict = {k.replace('trajectory_encoder.', ''): v 
                                     for k, v in state_dict.items() 
                                     if k.startswith('trajectory_encoder.')}
                    self.trajectory_encoder.load_state_dict(traj_state_dict, strict=False)
                    cxr_state_dict = {
                        k.replace('cxr_encoder.', ''): v
                        for k, v in state_dict.items()
                        if k.startswith('cxr_encoder.')
                    }
                    if cxr_state_dict:
                        self.cxr_encoder.load_state_dict(cxr_state_dict, strict=False)

            # Projection layer to convert abnormality features to UNet conditioning dimension
            self.abn_to_unet_proj = nn.Linear(
                self.abn_feat_dim * len(self.abnormality_names),
                77 * 768  # CLIP text encoder output dimension
            )

            self.train_adapter_only = config.get('train_adapter_only', False)
            self.freeze_unet_backbone = config.get('freeze_unet_backbone', False) or self.train_adapter_only
            self.base_learning_rate = config.get('learning_rate', 5e-5)
            self.adapter_learning_rate = config.get('adapter_learning_rate', self.base_learning_rate)
            self.unet_learning_rate = config.get('unet_learning_rate', self.base_learning_rate)
            self.ref_adapter = None

            if config.get('use_ref_adapter', True):
                self.ref_adapter = RefImageAdapter(
                    adapter_model_id=config.get('adapter_model_id'),
                    adapter_pretrained_path=config.get('adapter_pretrained_path'),
                    in_channels=config.get('adapter_in_channels', 3),
                    channels=config.get('adapter_channels', [320, 640, 1280, 1280]),
                    num_res_blocks=config.get('adapter_num_res_blocks', 2),
                    downscale_factor=config.get('adapter_downscale_factor', 8),
                    adapter_type=config.get('adapter_type', 'full_adapter'),
                    conditioning_scale=config.get('adapter_conditioning_scale', 1.0),
                )

                if config.get('freeze_ref_adapter', False):
                    for param in self.ref_adapter.parameters():
                        param.requires_grad = False

            if self.freeze_unet_backbone:
                for param in self.unet.parameters():
                    param.requires_grad = False
            
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
            
            self.scheduler = DDPMScheduler.from_pretrained(sd_root, subfolder="scheduler")

    def _build_sequence_mask(self, lab_lengths, seq_length, device):
        time_indices = torch.arange(seq_length, device=device).unsqueeze(0)
        return time_indices < lab_lengths.unsqueeze(1)

    def _get_ref_adapter_residuals(
        self,
        ref_img,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    ):
        if self.ref_adapter is None:
            return None

        return self.ref_adapter(
            ref_img,
            conditioning_scale=self.config.get('adapter_conditioning_scale', 1.0),
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

    def training_step(self, batch, batch_idx):
        if self.stage == 1:
            return self.stage1_training_step(batch, batch_idx)
        elif self.stage == 2:
            return self.stage2_training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        if self.stage == 1:
            return self.stage1_validation_step(batch, batch_idx, prefix="val")
        elif self.stage == 2:
            return self.stage2_validation_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        if self.stage == 1:
            return self.stage1_validation_step(batch, batch_idx, prefix="test")
        elif self.stage == 2:
            return self.stage2_validation_step(batch, batch_idx, prefix="test")

    def stage1_training_step(self, batch, batch_idx):
        # Unpack batch (dictionary)
        lab_test = batch['lab_test']
        lab_lengths = batch['lab_lengths']
        img_0 = batch['ref_img']
        img_1 = batch['target_img'] # (B, C, H, W)
        
        batch_size = lab_test.shape[0]
        seq_mask = self._build_sequence_mask(lab_lengths, lab_test.shape[1], lab_test.device)
        
        # [REFRACTOR] Construct target time indices (t_gen)
        # Use the last valid time step as the target time for training
        # t_gen should be (B, 1)
        t_gen = (lab_lengths - 1).view(batch_size, 1).to(lab_test.device)
        
        # Extract ground-truth imaging abnormality features from target image
        with torch.no_grad():
            # img_1 is (B, C, H, W)
            abn_features_gt = self.cxr_encoder(img_1)  # (B, num_abnormalities, abn_feat_dim)
        
        # Extract lab test features per time step
        lab_features = self.lab_test_encoder.encode_per_lab_test(lab_test, padding_masks=seq_mask)
        # lab_features: (batch_size, seq_length, num_lab_tests, lab_feat_dim)
        
        seq_length = lab_features.shape[1]
        
        # Transform lab test features to organ states
        organ_states_from_lab, organ_mask_from_lab = self.knowledge_transform.lab_to_organs(
            lab_features,
            self.lab_test_names,
            time_mask=seq_mask
        )
        
        # Extract abnormality features from img_0 (if available) and transform to organ states
        with torch.no_grad():
            abn_features_0 = self.cxr_encoder(img_0)  # (batch_size, num_abnormalities, abn_feat_dim)
        
        # Expand abnormality features to match time sequence
        abn_features_seq = torch.zeros(
            batch_size, seq_length, len(self.abnormality_names), self.abn_feat_dim,
            device=lab_test.device, dtype=abn_features_0.dtype
        )
        abn_features_seq[:, 0] = abn_features_0  # Place img_0 features at first time step
        
        # Place img_1 features at the correct target time step (t_gen)
        batch_indices = torch.arange(batch_size, device=lab_test.device)
        abn_features_seq[batch_indices, t_gen.squeeze(1)] = abn_features_gt
        
        # Transform abnormality features to organ states
        abn_time_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=lab_test.device)
        abn_time_mask[:, 0] = True  # img_0 available
        abn_time_mask[batch_indices, t_gen.squeeze(1)] = True  # img_1 available at t_gen
        
        organ_states_from_abn, organ_mask_from_abn = self.knowledge_transform.abn_to_organs(
            abn_features_seq,
            self.abnormality_names,
            time_mask=abn_time_mask
        )
        
        # Combine organ states from lab tests and abnormalities
        organ_mask = organ_mask_from_lab | organ_mask_from_abn
        organ_states = organ_states_from_lab.clone()
        organ_states[organ_mask_from_abn] = organ_states_from_abn[organ_mask_from_abn]
        
        # Build organ trajectory using trajectory encoder
        encoded_organ_states = self.trajectory_encoder(organ_states, organ_mask)
        
        # Infer organ states at generation time points
        # t_gen is (B, 1) -> infer_at_time returns (B, 1, num_organs, feat_dim)
        inferred_organ_states = self.trajectory_encoder.infer_at_time(
            encoded_organ_states,
            organ_mask,
            t_gen
        )
        
        # [REFRACTOR] Optimize: Squeeze the time dimension to match (B, num_organs, feat_dim)
        inferred_organ_states = inferred_organ_states.squeeze(1)
        
        # Construct gen_mask for the target time point (extract from organ_mask at t_gen)
        # organ_mask is (B, T, num_organs)
        gen_mask = organ_mask[batch_indices, t_gen.squeeze(1)] # (B, num_organs)
        
        # Transform to abnormality features
        inferred_abn_features = self.knowledge_transform.organs_to_abnormalities(
            inferred_organ_states,
            gen_mask,
            self.abnormality_names
        )
        # inferred_abn_features: (B, num_abnormalities, abn_feat_dim)
        
        # Compute MSE loss
        loss = F.mse_loss(inferred_abn_features, abn_features_gt)
        self.log(
            "train_loss_s1",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def stage1_validation_step(self, batch, batch_idx, prefix="val"):
        # Unpack batch
        lab_test = batch['lab_test']
        lab_lengths = batch['lab_lengths']
        img_0 = batch['ref_img']
        img_1 = batch['target_img']
        
        batch_size = lab_test.shape[0]
        seq_mask = self._build_sequence_mask(lab_lengths, lab_test.shape[1], lab_test.device)
        
        t_gen = (lab_lengths - 1).view(batch_size, 1).to(lab_test.device)
        
        with torch.no_grad():
            abn_features_gt = self.cxr_encoder(img_1)
        
        lab_features = self.lab_test_encoder.encode_per_lab_test(lab_test, padding_masks=seq_mask)
        seq_length = lab_features.shape[1]
        
        organ_states_from_lab, organ_mask_from_lab = self.knowledge_transform.lab_to_organs(
            lab_features,
            self.lab_test_names,
            time_mask=seq_mask
        )
        
        with torch.no_grad():
            abn_features_0 = self.cxr_encoder(img_0)
        
        abn_features_seq = torch.zeros(
            batch_size, seq_length, len(self.abnormality_names), self.abn_feat_dim,
            device=lab_test.device, dtype=abn_features_0.dtype
        )
        abn_features_seq[:, 0] = abn_features_0
        
        batch_indices = torch.arange(batch_size, device=lab_test.device)
        abn_features_seq[batch_indices, t_gen.squeeze(1)] = abn_features_gt
        
        abn_time_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=lab_test.device)
        abn_time_mask[:, 0] = True
        abn_time_mask[batch_indices, t_gen.squeeze(1)] = True
        
        organ_states_from_abn, organ_mask_from_abn = self.knowledge_transform.abn_to_organs(
            abn_features_seq,
            self.abnormality_names,
            time_mask=abn_time_mask
        )
        
        organ_mask = organ_mask_from_lab | organ_mask_from_abn
        organ_states = organ_states_from_lab.clone()
        organ_states[organ_mask_from_abn] = organ_states_from_abn[organ_mask_from_abn]
        
        encoded_organ_states = self.trajectory_encoder(organ_states, organ_mask)
        
        inferred_organ_states = self.trajectory_encoder.infer_at_time(
            encoded_organ_states,
            organ_mask,
            t_gen
        ).squeeze(1)
        
        gen_mask = organ_mask[batch_indices, t_gen.squeeze(1)]
        
        inferred_abn_features = self.knowledge_transform.organs_to_abnormalities(
            inferred_organ_states,
            gen_mask,
            self.abnormality_names
        )
        
        loss = F.mse_loss(inferred_abn_features, abn_features_gt)
        self.log(
            f"{prefix}_loss_s1",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

    def stage2_training_step(self, batch, batch_idx):
        # Unpack batch
        lab_test = batch['lab_test']
        lab_lengths = batch['lab_lengths']
        img_0 = batch['ref_img']
        img_1 = batch['target_img'] # (B, C, H, W)
        
        batch_size = lab_test.shape[0]
        seq_mask = self._build_sequence_mask(lab_lengths, lab_test.shape[1], lab_test.device)
        
        t_gen = (lab_lengths - 1).view(batch_size, 1).to(lab_test.device)
        
        # Encode target images (img_1) to latents
        # img_1 is (B, C, H, W)
        latents_1 = self.vae.encode(img_1).latent_dist.sample() * self.vae.config.scaling_factor
        # latents_1: (B, C_latent, H_latent, W_latent)
        
        noise = torch.randn_like(latents_1)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()
        noisy_latents = self.scheduler.add_noise(latents_1, noise, timesteps)
        
        # Infer imaging abnormality features using trajectory module
        lab_features = self.lab_test_encoder.encode_per_lab_test(lab_test, padding_masks=seq_mask)
        seq_length = lab_features.shape[1]
        
        organ_states_from_lab, organ_mask_from_lab = self.knowledge_transform.lab_to_organs(
            lab_features,
            self.lab_test_names,
            time_mask=seq_mask
        )
        
        with torch.no_grad():
            abn_features_0 = self.cxr_encoder(img_0)
        
        abn_features_seq = torch.zeros(
            batch_size, seq_length, len(self.abnormality_names), self.abn_feat_dim,
            device=lab_test.device, dtype=abn_features_0.dtype
        )
        abn_features_seq[:, 0] = abn_features_0
        
        abn_time_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=lab_test.device)
        abn_time_mask[:, 0] = True
        
        organ_states_from_abn, organ_mask_from_abn = self.knowledge_transform.abn_to_organs(
            abn_features_seq,
            self.abnormality_names,
            time_mask=abn_time_mask
        )
        
        organ_mask = organ_mask_from_lab | organ_mask_from_abn
        organ_states = organ_states_from_lab.clone()
        organ_states[organ_mask_from_abn] = organ_states_from_abn[organ_mask_from_abn]
        
        encoded_organ_states = self.trajectory_encoder(organ_states, organ_mask)

        # Infer organ states at generation time points
        inferred_organ_states = self.trajectory_encoder.infer_at_time(
            encoded_organ_states,
            organ_mask,
            t_gen
        ).squeeze(1) # (B, num_organs, feat_dim)
        
        batch_indices = torch.arange(batch_size, device=lab_test.device)
        gen_mask = organ_mask[batch_indices, t_gen.squeeze(1)]
        
        inferred_abn_features = self.knowledge_transform.organs_to_abnormalities(
            inferred_organ_states,
            gen_mask,
            self.abnormality_names
        )
        # inferred_abn_features: (B, num_abnormalities, abn_feat_dim)
        
        # Project abnormality features to UNet conditioning dimension
        abn_features_flat = inferred_abn_features.view(batch_size, -1)
        unet_conditioning = self.abn_to_unet_proj(abn_features_flat)
        unet_conditioning = unet_conditioning.view(batch_size, 77, 768)

        adapter_residuals = self._get_ref_adapter_residuals(img_0)
        unet_kwargs = {"encoder_hidden_states": unet_conditioning}
        if adapter_residuals is not None:
            unet_kwargs["down_intrablock_additional_residuals"] = [state.clone() for state in adapter_residuals]

        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, **unet_kwargs).sample
        
        loss = F.mse_loss(noise_pred, noise)
        self.log(
            "train_loss_s2",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def stage2_validation_step(self, batch, batch_idx, prefix="val"):
        # Unpack batch
        lab_test = batch['lab_test']
        lab_lengths = batch['lab_lengths']
        img_0 = batch['ref_img']
        img_1 = batch['target_img']
        
        batch_size = lab_test.shape[0]
        seq_mask = self._build_sequence_mask(lab_lengths, lab_test.shape[1], lab_test.device)
        
        t_gen = (lab_lengths - 1).view(batch_size, 1).to(lab_test.device)
        
        latents_1 = self.vae.encode(img_1).latent_dist.sample() * self.vae.config.scaling_factor
        
        noise = torch.randn_like(latents_1)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()
        noisy_latents = self.scheduler.add_noise(latents_1, noise, timesteps)
        
        lab_features = self.lab_test_encoder.encode_per_lab_test(lab_test, padding_masks=seq_mask)
        seq_length = lab_features.shape[1]
        
        organ_states_from_lab, organ_mask_from_lab = self.knowledge_transform.lab_to_organs(
            lab_features,
            self.lab_test_names,
            time_mask=seq_mask
        )
        
        with torch.no_grad():
            abn_features_0 = self.cxr_encoder(img_0)
        
        abn_features_seq = torch.zeros(
            batch_size, seq_length, len(self.abnormality_names), self.abn_feat_dim,
            device=lab_test.device, dtype=abn_features_0.dtype
        )
        abn_features_seq[:, 0] = abn_features_0
        
        abn_time_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=lab_test.device)
        abn_time_mask[:, 0] = True
        
        organ_states_from_abn, organ_mask_from_abn = self.knowledge_transform.abn_to_organs(
            abn_features_seq,
            self.abnormality_names,
            time_mask=abn_time_mask
        )
        
        organ_mask = organ_mask_from_lab | organ_mask_from_abn
        organ_states = organ_states_from_lab.clone()
        organ_states[organ_mask_from_abn] = organ_states_from_abn[organ_mask_from_abn]
        
        encoded_organ_states = self.trajectory_encoder(organ_states, organ_mask)
        
        inferred_organ_states = self.trajectory_encoder.infer_at_time(
            encoded_organ_states,
            organ_mask,
            t_gen
        ).squeeze(1)
        
        batch_indices = torch.arange(batch_size, device=lab_test.device)
        gen_mask = organ_mask[batch_indices, t_gen.squeeze(1)]
        
        inferred_abn_features = self.knowledge_transform.organs_to_abnormalities(
            inferred_organ_states,
            gen_mask,
            self.abnormality_names
        )
        
        abn_features_flat = inferred_abn_features.view(batch_size, -1)
        unet_conditioning = self.abn_to_unet_proj(abn_features_flat)
        unet_conditioning = unet_conditioning.view(batch_size, 77, 768)

        adapter_residuals = self._get_ref_adapter_residuals(img_0)
        unet_kwargs = {"encoder_hidden_states": unet_conditioning}
        if adapter_residuals is not None:
            unet_kwargs["down_intrablock_additional_residuals"] = [state.clone() for state in adapter_residuals]

        noise_pred = self.unet(noisy_latents, timesteps, **unet_kwargs).sample
        
        loss = F.mse_loss(noise_pred, noise)
        self.log(
            f"{prefix}_loss_s2",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        if batch_idx == 0 and prefix == "val":
            # Generate and save a sample image (only during validation to save time/space)
            with torch.no_grad():
                sample_latents = torch.randn_like(latents_1)
                sample_kwargs = {"encoder_hidden_states": unet_conditioning}
                if adapter_residuals is not None:
                    sample_kwargs["down_intrablock_additional_residuals"] = [state.clone() for state in adapter_residuals]
                pipeline_output = self.unet(sample_latents, timesteps, **sample_kwargs).sample
                generated_latents = 1 / self.vae.config.scaling_factor * pipeline_output
                generated_image = self.vae.decode(generated_latents).sample
                save_image(generated_image, os.path.join(self.config['output_dir'], f"epoch_{self.current_epoch}.png"))

    def configure_optimizers(self):
        if self.stage == 1:
            params_to_train = (
                list(self.lab_test_encoder.parameters()) +
                list(self.knowledge_transform.parameters()) +
                list(self.trajectory_encoder.parameters())
            )
            params_to_train = [param for param in params_to_train if param.requires_grad]
            optimizer = torch.optim.AdamW(params_to_train, lr=self.config['learning_rate'])
            return optimizer
        elif self.stage == 2:
            optimizer_groups = []

            backbone_params = [
                param for param in (
                    list(self.lab_test_encoder.parameters()) +
                    list(self.knowledge_transform.parameters()) +
                    list(self.trajectory_encoder.parameters()) +
                    list(self.abn_to_unet_proj.parameters())
                )
                if param.requires_grad
            ]
            if backbone_params:
                optimizer_groups.append({
                    "params": backbone_params,
                    "lr": self.base_learning_rate,
                })

            if self.ref_adapter is not None:
                adapter_params = [param for param in self.ref_adapter.parameters() if param.requires_grad]
                if adapter_params:
                    optimizer_groups.append({
                        "params": adapter_params,
                        "lr": self.adapter_learning_rate,
                    })

            if not self.freeze_unet_backbone:
                unet_params = [param for param in self.unet.parameters() if param.requires_grad]
                if unet_params:
                    optimizer_groups.append({
                        "params": unet_params,
                        "lr": self.unet_learning_rate,
                    })

            if not optimizer_groups:
                raise ValueError("No trainable parameters found for Stage 2 optimizer.")

            return torch.optim.AdamW(optimizer_groups)

def main(config):
    pl.seed_everything(config['seed'])
    
    dm = ControlGenDataModule(config)
    model = ControlGenModel(config)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config['output_dir'],
        filename='best_model',
        save_top_k=1,
        monitor=f"val_loss_s{config['stage']}_epoch",
        mode='min',
    )

    # Keep finished epoch bars in terminal so metric history is visible.
    progress_bar_callback = pl.callbacks.TQDMProgressBar(
        refresh_rate=config.get('progress_bar_refresh_rate', 1),
        leave=config.get('progress_bar_leave', True),
    )
    
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=config['epochs'],
        callbacks=[checkpoint_callback, progress_bar_callback],
        logger=pl.loggers.TensorBoardLogger(save_dir=config['log_dir']),
    )
    
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    main(config)
