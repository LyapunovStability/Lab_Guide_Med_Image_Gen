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

        # Static graph edges for vectorized lab/org/abn transforms (no Python loops over batch/time)
        self._register_graph_edge_buffers()

    def _relation_embedding_vector(self, relation: str) -> torch.Tensor:
        """Single relation row; zeros if unknown (matches legacy _get_relation_embedding)."""
        z = torch.zeros(self.concept_emb_dim, dtype=torch.float32)
        if hasattr(self, "relation_to_idx") and relation in self.relation_to_idx:
            idx = self.relation_to_idx[relation]
            z = self.relation_emb[idx].detach().float().cpu()
        return z

    def _register_graph_edge_buffers(self) -> None:
        """Precompute organ↔lab and organ↔abnormality edges for batched forward."""
        lab_rows = self.organ_graph.get_all_lab_tests()
        lab_name_to_feat_idx = {n: i for i, n in enumerate(lab_rows)}
        abn_rows = self.organ_graph.get_all_abnormalities()

        lab_idx_list: List[int] = []
        org_idx_list: List[int] = []
        rel_vecs: List[torch.Tensor] = []

        for org_idx, org_name in enumerate(self.organ_graph.get_all_organs()):
            for lab_name in self.organ_graph.get_labs_for_organ(org_name):
                li = lab_name_to_feat_idx.get(lab_name, -1)
                if li < 0:
                    continue
                triplets = self.organ_graph.get_lab_triplets(lab_name)
                if triplets:
                    _, relation, _ = triplets[0]
                    rvec = self._relation_embedding_vector(relation)
                else:
                    rvec = torch.zeros(self.concept_emb_dim, dtype=torch.float32)
                lab_idx_list.append(li)
                org_idx_list.append(org_idx)
                rel_vecs.append(rvec)

        if lab_idx_list:
            self.register_buffer(
                "lab_org_lab_idx",
                torch.tensor(lab_idx_list, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "lab_org_org_idx",
                torch.tensor(org_idx_list, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "lab_org_rel_emb",
                torch.stack(rel_vecs, dim=0),
                persistent=False,
            )
        else:
            self.register_buffer("lab_org_lab_idx", torch.zeros(0, dtype=torch.long), persistent=False)
            self.register_buffer("lab_org_org_idx", torch.zeros(0, dtype=torch.long), persistent=False)
            self.register_buffer(
                "lab_org_rel_emb",
                torch.zeros(0, self.concept_emb_dim, dtype=torch.float32),
                persistent=False,
            )

        abn_idx_list: List[int] = []
        abn_org_idx_list: List[int] = []
        abn_rel_vecs: List[torch.Tensor] = []

        for org_idx, org_name in enumerate(self.organ_graph.get_all_organs()):
            for abn_name in self.organ_graph.get_abnormalities_for_organ(org_name):
                ai = self.organ_graph.get_abnormality_index(abn_name)
                if ai < 0:
                    continue
                triplets = self.organ_graph.get_abnormality_triplets(abn_name)
                if triplets:
                    _, relation, _ = triplets[0]
                    rvec = self._relation_embedding_vector(relation)
                else:
                    rvec = torch.zeros(self.concept_emb_dim, dtype=torch.float32)
                abn_idx_list.append(ai)
                abn_org_idx_list.append(org_idx)
                abn_rel_vecs.append(rvec)

        if abn_idx_list:
            self.register_buffer(
                "abn_org_abn_idx",
                torch.tensor(abn_idx_list, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "abn_org_org_idx",
                torch.tensor(abn_org_idx_list, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "abn_org_rel_emb",
                torch.stack(abn_rel_vecs, dim=0),
                persistent=False,
            )
        else:
            self.register_buffer("abn_org_abn_idx", torch.zeros(0, dtype=torch.long), persistent=False)
            self.register_buffer("abn_org_org_idx", torch.zeros(0, dtype=torch.long), persistent=False)
            self.register_buffer(
                "abn_org_rel_emb",
                torch.zeros(0, self.concept_emb_dim, dtype=torch.float32),
                persistent=False,
            )

        o2a_abn_idx: List[int] = []
        o2a_org_idx: List[int] = []
        for abn_i, abn_name in enumerate(abn_rows):
            for org_name in self.organ_graph.get_organs_for_abnormality(abn_name):
                oi = self.organ_graph.get_organ_index(org_name)
                if oi >= 0:
                    o2a_abn_idx.append(abn_i)
                    o2a_org_idx.append(oi)

        if o2a_abn_idx:
            self.register_buffer(
                "o2a_abn_idx",
                torch.tensor(o2a_abn_idx, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "o2a_org_idx",
                torch.tensor(o2a_org_idx, dtype=torch.long),
                persistent=False,
            )
        else:
            self.register_buffer("o2a_abn_idx", torch.zeros(0, dtype=torch.long), persistent=False)
            self.register_buffer("o2a_org_idx", torch.zeros(0, dtype=torch.long), persistent=False)

    def _scatter_mean_to_organs(
        self,
        attended: torch.Tensor,
        org_indices: torch.Tensor,
        num_organs: int,
        time_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        attended: (B, T, E, F), org_indices: (E,) long on same device.
        time_mask: (B, T) bool, True = valid step; counts only apply when valid (legacy loop behavior).
        Returns per-organ mean pre-activations (B, T, O, F) and organ_mask (B, T, O).
        """
        if attended.shape[2] == 0:
            b, t, _, f = attended.shape
            z = attended.new_zeros(b, t, num_organs, f)
            m = torch.zeros(b, t, num_organs, dtype=torch.bool, device=attended.device)
            return z, m

        b, t, e, f = attended.shape
        flat = b * t
        att = attended.reshape(flat, e, f)
        sum_o = attended.new_zeros(flat, num_organs, f)
        cnt = attended.new_zeros(flat, num_organs, 1)
        oi = org_indices.to(device=attended.device, dtype=torch.long)
        if time_mask is not None:
            w = time_mask.reshape(flat).to(dtype=att.dtype).unsqueeze(-1)
        else:
            w = torch.ones(flat, 1, device=attended.device, dtype=att.dtype)
        for ei in range(e):
            o = int(oi[ei])
            sum_o[:, o] = sum_o[:, o] + att[:, ei]
            cnt[:, o] = cnt[:, o] + w
        mean_o = sum_o / cnt.clamp(min=1e-9)
        mask = (cnt.squeeze(-1) > 0).reshape(b, t, num_organs)
        return mean_o.reshape(b, t, num_organs, f), mask

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
        e = int(self.lab_org_lab_idx.shape[0])
        if e == 0:
            z = lab_features.new_zeros(
                batch_size, num_time_steps, num_organs, self.organ_feat_dim
            )
            m = torch.zeros(
                batch_size, num_time_steps, num_organs,
                dtype=torch.bool, device=lab_features.device,
            )
            return z, m

        graph_labs = self.organ_graph.get_all_lab_tests()
        if len(lab_names) != len(graph_labs) or any(a != b for a, b in zip(lab_names, graph_labs)):
            raise ValueError(
                "lab_to_organs (vectorized): lab_names must match organ_graph.get_all_lab_tests() order."
            )

        li = self.lab_org_lab_idx.to(lab_features.device)
        lab_e = lab_features[:, :, li, :]
        lc = self.lab_concept_emb[li].to(device=lab_features.device, dtype=lab_features.dtype)
        lc = lc.unsqueeze(0).unsqueeze(0).expand(batch_size, num_time_steps, -1, -1)
        fused = torch.cat([lab_e, lc], dim=-1)
        fused_flat = fused.reshape(-1, fused.shape[-1])
        h_lab = F.relu(self.W_lab(fused_flat)).view(batch_size, num_time_steps, e, self.lab_feat_dim)

        rel_e = self.lab_org_rel_emb.to(device=lab_features.device, dtype=lab_features.dtype)
        dr = self.D(rel_e)
        alpha = torch.sigmoid((h_lab * dr.unsqueeze(0).unsqueeze(0)).sum(dim=-1, keepdim=True))
        attended = h_lab * alpha

        if time_mask is not None:
            attended = attended * time_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=attended.dtype)

        h_agg, organ_mask = self._scatter_mean_to_organs(
            attended, self.lab_org_org_idx, num_organs, time_mask
        )
        h_org = F.relu(self.W_org(h_agg.reshape(-1, self.lab_feat_dim))).view(
            batch_size, num_time_steps, num_organs, self.organ_feat_dim
        )
        organ_states = torch.where(
            organ_mask.unsqueeze(-1),
            h_org,
            torch.zeros_like(h_org),
        )
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
        e = int(self.abn_org_abn_idx.shape[0])
        if e == 0:
            z = abn_features.new_zeros(
                batch_size, num_time_steps, num_organs, self.organ_feat_dim
            )
            m = torch.zeros(
                batch_size, num_time_steps, num_organs,
                dtype=torch.bool, device=abn_features.device,
            )
            return z, m

        graph_abns = self.organ_graph.get_all_abnormalities()
        if len(abn_names) != len(graph_abns) or any(a != b for a, b in zip(abn_names, graph_abns)):
            raise ValueError(
                "abn_to_organs (vectorized): abn_names must match organ_graph.get_all_abnormalities() order."
            )

        ai = self.abn_org_abn_idx.to(abn_features.device)
        abn_e = abn_features[:, :, ai, :]
        ac = self.abn_concept_emb[ai].to(device=abn_features.device, dtype=abn_features.dtype)
        ac = ac.unsqueeze(0).unsqueeze(0).expand(batch_size, num_time_steps, -1, -1)
        fused = torch.cat([abn_e, ac], dim=-1)
        fused_flat = fused.reshape(-1, fused.shape[-1])
        h_abn = F.relu(self.W_abn(fused_flat)).view(batch_size, num_time_steps, e, self.abn_feat_dim)

        rel_e = self.abn_org_rel_emb.to(device=abn_features.device, dtype=abn_features.dtype)
        dr = self.D(rel_e)
        alpha = torch.sigmoid((h_abn * dr.unsqueeze(0).unsqueeze(0)).sum(dim=-1, keepdim=True))
        attended = h_abn * alpha

        if time_mask is not None:
            attended = attended * time_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=attended.dtype)

        h_agg, organ_mask = self._scatter_mean_to_organs(
            attended, self.abn_org_org_idx, num_organs, time_mask
        )
        h_org = F.relu(self.W_org(h_agg.reshape(-1, self.lab_feat_dim))).view(
            batch_size, num_time_steps, num_organs, self.organ_feat_dim
        )
        organ_states = torch.where(
            organ_mask.unsqueeze(-1),
            h_org,
            torch.zeros_like(h_org),
        )
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
        graph_abns = self.organ_graph.get_all_abnormalities()
        if len(abn_names) != len(graph_abns) or any(a != b for a, b in zip(abn_names, graph_abns)):
            raise ValueError(
                "organs_to_abnormalities (vectorized): abn_names must match organ_graph.get_all_abnormalities() order."
            )

        e = int(self.o2a_org_idx.shape[0])
        if e == 0:
            abn_features = organ_states.new_zeros(
                batch_size, num_time_steps, num_abns, self.abn_feat_dim
            )
        else:
            oi = self.o2a_org_idx.to(organ_states.device)
            ai = self.o2a_abn_idx.to(organ_states.device)
            org_e = organ_states[:, :, oi, :]
            m_e = organ_mask[:, :, oi].to(dtype=org_e.dtype)
            org_e = org_e * m_e.unsqueeze(-1)
            w_edge = m_e

            flat = batch_size * num_time_steps
            att = org_e.reshape(flat, e, self.organ_feat_dim)
            w_flat = w_edge.reshape(flat, e)
            sum_a = organ_states.new_zeros(flat, num_abns, self.organ_feat_dim)
            cnt_a = organ_states.new_zeros(flat, num_abns, 1)
            for ei in range(e):
                a = int(ai[ei])
                sum_a[:, a] = sum_a[:, a] + att[:, ei]
                cnt_a[:, a] = cnt_a[:, a] + w_flat[:, ei].unsqueeze(-1)
            h_agg = sum_a / cnt_a.clamp(min=1e-9)
            h_abn = F.relu(self.W_org_to_abn(h_agg.reshape(-1, self.organ_feat_dim))).view(
                batch_size, num_time_steps, num_abns, self.abn_feat_dim
            )
            has_any = (cnt_a.squeeze(-1) > 0).reshape(batch_size, num_time_steps, num_abns)
            abn_features = torch.where(
                has_any.unsqueeze(-1),
                h_abn,
                torch.zeros_like(h_abn),
            )

        if squeezed_time_dim:
            return abn_features.squeeze(1)

        return abn_features

