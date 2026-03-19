import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import os

class CXR_Encoder(pl.LightningModule):
    def __init__(self,
                 task='na',
                 vision_backbone='convnext_base',
                 pretrained=True,
                 num_abnormalities=12,
                 feature_dim=512,
                 eva_x_weights_path=None,
                 eva_x_model_path=None,
                 ):
        
        super().__init__()

        self.task = task
        self.num_abnormalities = num_abnormalities
        self.feature_dim = feature_dim
        
        # Check if using EVA-X model
        is_eva_x = vision_backbone.startswith('eva02_') or vision_backbone.startswith('eva_x')
        
        if is_eva_x:
            # EVA-X models are stored in utils directory (external code from EVA-X project)
            # Import models_eva to register models with timm
            try:
                import utils.models_eva as models_eva  # This registers the models with timm
            except ImportError:
                raise ImportError("Could not import utils.models_eva. Make sure models_eva.py is in the utils directory.")
            
            # Prepare kwargs for EVA-X initialization
            eva_kwargs = {
                'num_classes': 0,  # Remove classifier head
                'img_size': 512,   # Match input image size
                'use_mean_pooling': True,  # Get pooled features
                'in_chans': 3
            }
            
            # Add init_ckpt if pretrained weights are provided
            if pretrained and eva_x_weights_path is not None:
                if os.path.exists(eva_x_weights_path):
                    eva_kwargs['init_ckpt'] = eva_x_weights_path
                else:
                    print(f"Warning: EVA-X weights path not found: {eva_x_weights_path}")
                    pretrained = False
            
            # Create EVA-X model
            self.vision_backbone = timm.create_model(
                vision_backbone,
                pretrained=pretrained,
                **eva_kwargs
            )
        else:
            # Standard timm model initialization
            self.vision_backbone = timm.create_model(
                vision_backbone,
                pretrained=pretrained,
                num_classes=0,  # Set num_classes to 0 to remove the classifier head
                in_chans=3
            )
        
        # Get the feature dimension from the backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 512, 512)
            backbone_feat = self.vision_backbone(dummy_input)
            backbone_dim = backbone_feat.shape[1] if len(backbone_feat.shape) > 1 else backbone_feat.shape[-1]
        
        # Projection layer to map backbone features to abnormality features
        # Output shape: (batch_size, num_abnormalities, feature_dim)
        self.abnormality_projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim * num_abnormalities),
            nn.ReLU(),
            nn.Linear(feature_dim * num_abnormalities, num_abnormalities * feature_dim)
        )
        
    def forward(self, x):
        """
        Extract imaging abnormality features from CXR images.
        
        Args:
            x: Input images of shape (batch_size, 3, 512, 512)
        
        Returns:
            features: Imaging abnormality features of shape (batch_size, num_abnormalities, feature_dim)
        """
        # Extract backbone features
        backbone_feat = self.vision_backbone(x)
        
        # Handle different output shapes from different backbones
        if len(backbone_feat.shape) == 4:
            # If output is (B, C, H, W), global average pool
            backbone_feat = torch.mean(backbone_feat, dim=(2, 3))
        elif len(backbone_feat.shape) == 3:
            # If output is (B, N, C), take mean over sequence dimension
            backbone_feat = torch.mean(backbone_feat, dim=1)
        
        # Project to abnormality features
        flat_features = self.abnormality_projection(backbone_feat)
        
        # Reshape to (batch_size, num_abnormalities, feature_dim)
        features = flat_features.view(backbone_feat.shape[0], self.num_abnormalities, self.feature_dim)
        
        return features