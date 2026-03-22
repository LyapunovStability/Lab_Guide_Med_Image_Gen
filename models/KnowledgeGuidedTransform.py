import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
from models.OrganGraph import OrganGraph
from utils.project_paths import resolve_pubmedbert_source

class KnowledgeGuidedTransform(nn.Module):
    """
    Knowledge-Guided Feature Transformation Module
    
    Transforms features between laboratory tests, imaging abnormalities, and organs
    using graph-based information passing with concept embeddings.
    """
    
    def __init__(
        self,
        organ_graph: OrganGraph,
        lab_feat_dim: int,
        abn_feat_dim: int,
        organ_feat_dim: int,
        concept_emb_dim: int = 768,
        use_clip: bool = True,
        text_encoder_model_name: Optional[str] = None,
        text_max_length: int = 512,
        device: str = 'cuda'
    ):
        """
        Initialize the knowledge-guided transformation module.
        
        Args:
            organ_graph: OrganGraph instance
            lab_feat_dim: Dimension of lab test value features
            abn_feat_dim: Dimension of imaging abnormality features
            organ_feat_dim: Dimension of organ state features
            concept_emb_dim: Dimension of concept embeddings (default: 768)
            use_clip: Backward-compatible flag for enabling pretrained text encoder
            text_encoder_model_name: 本地目录（相对项目根或绝对路径）或 Hub 模型 ID；None 则默认 checkpoints/pubmedbert-base-embeddings
            text_max_length: Maximum token length for concept text encoding
            device: Device to run on
        """
        super().__init__()
        
        self.organ_graph = organ_graph
        self.lab_feat_dim = lab_feat_dim
        self.abn_feat_dim = abn_feat_dim
        self.organ_feat_dim = organ_feat_dim
        self.concept_emb_dim = concept_emb_dim
        if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device
        self.text_max_length = text_max_length
        
        # Initialize pretrained text encoder for concept embeddings
        if use_clip:
            resolved_te = resolve_pubmedbert_source(text_encoder_model_name)
            self.text_encoder_model_name = resolved_te
            self.text_encoder = AutoModel.from_pretrained(resolved_te)
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_te)

            encoder_hidden_size = getattr(self.text_encoder.config, "hidden_size", None)
            if encoder_hidden_size is None:
                raise ValueError(
                    f"Text encoder `{self.text_encoder_model_name}` does not expose `hidden_size`."
                )
            if encoder_hidden_size != self.concept_emb_dim:
                raise ValueError(
                    f"Concept embedding dimension mismatch: model hidden size is {encoder_hidden_size}, "
                    f"but concept_emb_dim is {self.concept_emb_dim}."
                )

            self.text_encoder.to(self.device)
            # Freeze text encoder
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()
        else:
            self.text_encoder_model_name = text_encoder_model_name
            self.text_encoder = None
            self.tokenizer = None
        
        # Pre-compute concept embeddings for all entities
        self._precompute_concept_embeddings()
        
        # Projection matrices
        self.W_lab = nn.Linear(lab_feat_dim + concept_emb_dim, lab_feat_dim)
        self.W_abn = nn.Linear(abn_feat_dim + concept_emb_dim, abn_feat_dim)
        self.W_org = nn.Linear(lab_feat_dim, organ_feat_dim)  # For lab->org transformation
        
        # Attention weight matrix
        self.D = nn.Linear(concept_emb_dim, lab_feat_dim, bias=False)
        
        # Projection for organ->abnormality transformation
        self.W_org_to_abn = nn.Linear(organ_feat_dim, abn_feat_dim)
        
    def _precompute_concept_embeddings(self):
        """Pre-compute concept embeddings for all lab tests, organs, and abnormalities."""
        if self.text_encoder is None:
            # Use learnable embeddings if no pretrained text encoder is used
            num_labs = self.organ_graph.get_num_lab_tests()
            num_abns = self.organ_graph.get_num_abnormalities()
            num_orgs = self.organ_graph.get_num_organs()
            
            self.lab_concept_emb = nn.Parameter(
                torch.randn(num_labs, self.concept_emb_dim)
            )
            self.abn_concept_emb = nn.Parameter(
                torch.randn(num_abns, self.concept_emb_dim)
            )
            self.org_concept_emb = nn.Parameter(
                torch.randn(num_orgs, self.concept_emb_dim)
            )
            self.relation_emb = nn.Parameter(
                torch.randn(100, self.concept_emb_dim)  # Max 100 unique relations
            )
            return
        
        # Use pretrained text encoder to encode concepts
        lab_tests = self.organ_graph.get_all_lab_tests()
        abnormalities = self.organ_graph.get_all_abnormalities()
        organs = self.organ_graph.get_all_organs()
        
        # Encode lab tests
        lab_texts = [f"laboratory test: {lab}" for lab in lab_tests]
        lab_embeddings = self._encode_texts(lab_texts)
        self.register_buffer('lab_concept_emb', lab_embeddings)
        
        # Encode abnormalities
        abn_texts = [f"imaging abnormality: {abn}" for abn in abnormalities]
        abn_embeddings = self._encode_texts(abn_texts)
        self.register_buffer('abn_concept_emb', abn_embeddings)
        
        # Encode organs
        org_texts = [f"organ: {org}" for org in organs]
        org_embeddings = self._encode_texts(org_texts)
        self.register_buffer('org_concept_emb', org_embeddings)
        
        # Collect all unique relations and encode them
        all_relations = set()
        for lab in lab_tests:
            triplets = self.organ_graph.get_lab_triplets(lab)
            for _, rel, _ in triplets:
                all_relations.add(rel)
        for abn in abnormalities:
            triplets = self.organ_graph.get_abnormality_triplets(abn)
            for _, rel, _ in triplets:
                all_relations.add(rel)
        
        relation_list = sorted(list(all_relations))
        rel_texts = [f"relation: {rel}" for rel in relation_list]
        rel_embeddings = self._encode_texts(rel_texts)
        
        # Create mapping from relation string to index
        self.relation_to_idx = {rel: idx for idx, rel in enumerate(relation_list)}
        self.register_buffer('relation_emb', rel_embeddings)
    
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using pretrained text encoder."""
        if self.text_encoder is None:
            return torch.zeros(len(texts), self.concept_emb_dim)
        
        with torch.no_grad():
            tokenizer_max_len = getattr(self.tokenizer, "model_max_length", self.text_max_length)
            max_length = self.text_max_length
            if isinstance(tokenizer_max_len, int) and 0 < tokenizer_max_len < 100000:
                max_length = min(max_length, tokenizer_max_len)

            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.text_encoder(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        
        return embeddings
    
    def _get_concept_embedding(self, entity_name: str, entity_type: str, device: torch.device) -> torch.Tensor:
        """Get concept embedding for an entity."""
        if entity_type == 'lab':
            idx = self.organ_graph.get_lab_test_index(entity_name)
            if idx >= 0:
                return self.lab_concept_emb[idx].to(device)
        elif entity_type == 'abn':
            idx = self.organ_graph.get_abnormality_index(entity_name)
            if idx >= 0:
                return self.abn_concept_emb[idx].to(device)
        elif entity_type == 'org':
            idx = self.organ_graph.get_organ_index(entity_name)
            if idx >= 0:
                return self.org_concept_emb[idx].to(device)
        
        # Return zero embedding if not found
        return torch.zeros(self.concept_emb_dim, device=device)
    
    def _get_relation_embedding(self, relation: str, device: torch.device) -> torch.Tensor:
        """Get relation embedding."""
        if hasattr(self, 'relation_to_idx') and relation in self.relation_to_idx:
            idx = self.relation_to_idx[relation]
            return self.relation_emb[idx].to(device)
        
        # Return zero embedding if not found
        return torch.zeros(self.concept_emb_dim, device=device)
    
    def lab_to_organs(
        self,
        lab_features: torch.Tensor,
        lab_names: List[str],
        time_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform lab test features to organ states.
        
        Args:
            lab_features: (batch_size, num_time_steps, num_lab_tests, lab_feat_dim)
            lab_names: List of lab test names corresponding to feature dimensions
            time_mask: (batch_size, num_time_steps) mask for valid time steps
        
        Returns:
            organ_states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
            organ_mask: (batch_size, num_time_steps, num_organs) binary mask
        """
        batch_size, num_time_steps, num_labs, feat_dim = lab_features.shape
        num_organs = self.organ_graph.get_num_organs()
        
        # Initialize organ states and mask
        organ_states = torch.zeros(
            batch_size, num_time_steps, num_organs, self.organ_feat_dim,
            device=lab_features.device, dtype=lab_features.dtype
        )
        organ_mask = torch.zeros(
            batch_size, num_time_steps, num_organs,
            device=lab_features.device, dtype=torch.bool
        )
        
        # Process each time step
        for t in range(num_time_steps):
            for b in range(batch_size):
                # Check if this time step is valid
                if time_mask is not None and not time_mask[b, t]:
                    continue
                
                # Process each organ
                for org_idx, org_name in enumerate(self.organ_graph.get_all_organs()):
                    # Get connected lab tests
                    connected_labs = self.organ_graph.get_labs_for_organ(org_name)
                    
                    if not connected_labs:
                        continue
                    
                    # Collect features and compute attention weights
                    attended_features = []
                    
                    for lab_name in connected_labs:
                        # Find lab index
                        lab_idx = lab_names.index(lab_name) if lab_name in lab_names else -1
                        if lab_idx < 0 or lab_idx >= num_labs:
                            continue
                        
                        # Get lab feature
                        lab_feat = lab_features[b, t, lab_idx]  # (lab_feat_dim,)
                        
                        # Get concept embeddings
                        lab_concept = self._get_concept_embedding(lab_name, 'lab', lab_features.device)
                        org_concept = self._get_concept_embedding(org_name, 'org', lab_features.device)
                        
                        # Get relation embedding (use first relation found)
                        triplets = self.organ_graph.get_lab_triplets(lab_name)
                        if triplets:
                            _, relation, _ = triplets[0]
                            rel_emb = self._get_relation_embedding(relation, lab_features.device)
                        else:
                            rel_emb = torch.zeros(self.concept_emb_dim, device=lab_features.device)
                        
                        # Fuse value feature and concept embedding
                        fused_feat = torch.cat([lab_feat, lab_concept], dim=0)
                        h_lab = F.relu(self.W_lab(fused_feat))
                        
                        # Compute attention weight
                        alpha = torch.sigmoid(h_lab @ self.D(rel_emb))
                        
                        attended_features.append(alpha * h_lab)
                    
                    if attended_features:
                        # Aggregate attended features
                        h_agg = torch.stack(attended_features).mean(dim=0)
                        h_org = F.relu(self.W_org(h_agg))
                        
                        organ_states[b, t, org_idx] = h_org
                        organ_mask[b, t, org_idx] = True
        
        return organ_states, organ_mask
    
    def abn_to_organs(
        self,
        abn_features: torch.Tensor,
        abn_names: List[str],
        time_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform imaging abnormality features to organ states.
        
        Args:
            abn_features: (batch_size, num_time_steps, num_abnormalities, abn_feat_dim)
            abn_names: List of abnormality names
            time_mask: (batch_size, num_time_steps) mask for valid time steps
        
        Returns:
            organ_states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
            organ_mask: (batch_size, num_time_steps, num_organs) binary mask
        """
        batch_size, num_time_steps, num_abns, feat_dim = abn_features.shape
        num_organs = self.organ_graph.get_num_organs()
        
        # Initialize organ states and mask
        organ_states = torch.zeros(
            batch_size, num_time_steps, num_organs, self.organ_feat_dim,
            device=abn_features.device, dtype=abn_features.dtype
        )
        organ_mask = torch.zeros(
            batch_size, num_time_steps, num_organs,
            device=abn_features.device, dtype=torch.bool
        )
        
        # Process each time step
        for t in range(num_time_steps):
            for b in range(batch_size):
                if time_mask is not None and not time_mask[b, t]:
                    continue
                
                for org_idx, org_name in enumerate(self.organ_graph.get_all_organs()):
                    connected_abns = self.organ_graph.get_abnormalities_for_organ(org_name)
                    
                    if not connected_abns:
                        continue
                    
                    attended_features = []
                    
                    for abn_name in connected_abns:
                        abn_idx = abn_names.index(abn_name) if abn_name in abn_names else -1
                        if abn_idx < 0 or abn_idx >= num_abns:
                            continue
                        
                        abn_feat = abn_features[b, t, abn_idx]
                        
                        # Get concept embeddings
                        abn_concept = self._get_concept_embedding(abn_name, 'abn', abn_features.device)
                        org_concept = self._get_concept_embedding(org_name, 'org', abn_features.device)
                        
                        # Get relation embedding
                        triplets = self.organ_graph.get_abnormality_triplets(abn_name)
                        if triplets:
                            _, relation, _ = triplets[0]
                            rel_emb = self._get_relation_embedding(relation, abn_features.device)
                        else:
                            rel_emb = torch.zeros(self.concept_emb_dim, device=abn_features.device)
                        
                        # Fuse value feature and concept embedding
                        fused_feat = torch.cat([abn_feat, abn_concept], dim=0)
                        h_abn = F.relu(self.W_abn(fused_feat))
                        
                        # Compute attention weight
                        alpha = torch.sigmoid(h_abn @ self.D(rel_emb))
                        
                        attended_features.append(alpha * h_abn)
                    
                    if attended_features:
                        # Aggregate and project to organ dimension
                        h_agg = torch.stack(attended_features).mean(dim=0)
                        h_org = F.relu(self.W_org(h_agg))
                        
                        organ_states[b, t, org_idx] = h_org
                        organ_mask[b, t, org_idx] = True
        
        return organ_states, organ_mask
    
    def organs_to_abnormalities(
        self,
        organ_states: torch.Tensor,
        organ_mask: torch.Tensor,
        abn_names: List[str]
    ) -> torch.Tensor:
        """
        Transform organ states back to imaging abnormality features.
        
        Args:
            organ_states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
            organ_mask: (batch_size, num_time_steps, num_organs) binary mask
            abn_names: List of abnormality names to generate
        
        Returns:
            abn_features: (batch_size, num_time_steps, num_abnormalities, abn_feat_dim)
        """
        squeezed_time_dim = False
        if organ_states.dim() == 3:
            organ_states = organ_states.unsqueeze(1)
            squeezed_time_dim = True
        if organ_mask.dim() == 2:
            organ_mask = organ_mask.unsqueeze(1)

        batch_size, num_time_steps, num_organs, _ = organ_states.shape
        num_abns = len(abn_names)
        
        # Initialize abnormality features
        abn_features = torch.zeros(
            batch_size, num_time_steps, num_abns, self.abn_feat_dim,
            device=organ_states.device, dtype=organ_states.dtype
        )
        
        # Process each time step
        for t in range(num_time_steps):
            for b in range(batch_size):
                for abn_idx, abn_name in enumerate(abn_names):
                    # Get connected organs
                    connected_orgs = self.organ_graph.get_organs_for_abnormality(abn_name)
                    
                    if not connected_orgs:
                        continue
                    
                    # Aggregate organ states
                    org_features = []
                    for org_name in connected_orgs:
                        org_idx = self.organ_graph.get_organ_index(org_name)
                        if org_idx >= 0 and organ_mask[b, t, org_idx]:
                            org_features.append(organ_states[b, t, org_idx])
                    
                    if org_features:
                        # Aggregate and project to abnormality dimension
                        h_agg = torch.stack(org_features).mean(dim=0)
                        h_abn = F.relu(self.W_org_to_abn(h_agg))
                        abn_features[b, t, abn_idx] = h_abn

        if squeezed_time_dim:
            return abn_features.squeeze(1)

        return abn_features

