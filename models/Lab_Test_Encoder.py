# Modified  From: https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py
from typing import Optional, Any
import math

import torch
import pytorch_lightning as pl
try:
    from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
except ImportError:
    # Lightning >=2.x removed EPOCH_OUTPUT; keep typing compatibility.
    from pytorch_lightning.utilities.types import STEP_OUTPUT
    EPOCH_OUTPUT = Any
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from einops import rearrange, repeat
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import f1_score,roc_auc_score, average_precision_score


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # `is_causal` is forwarded by newer torch TransformerEncoder API.
        _ = is_causal
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class Lab_Test_Encoder(pl.LightningModule):

    def __init__(self,
                 max_len,
                 d_model,
                 n_heads,
                 num_layers,
                 dim_feedforward,
                 feat_dim=79,
                 num_classes=1,
                 task='mortality',
                 pos_weight=30,
                 dropout=0.3,
                 pos_encoding='fixed',
                 activation='gelu', norm='BatchNorm',
                 freeze=False,
                 hid_dim_1=128,
                 max_epoches=100,
                 ignore_keys=[],
                 mode='min',
                 monitor=None,
                 pool='mean',
                 cond=False,
                 ckpt_path=None):
        # super(TSTransformerEncoder, self).__init__()
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.mode=mode
        self.max_epoches=max_epoches
        self.task = task

        pos_weight = torch.tensor([pos_weight])
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.num_classes = num_classes
        if task == 'los':
            self.loss = nn.CrossEntropyLoss()
            self.num_classes = 5

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len+1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # self.mlp_head = nn.Linear(d_model, self.num_classes)
        self.mlp_head = nn.Sequential(nn.Linear(d_model, hid_dim_1),
                                      nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(hid_dim_1, self.num_classes)
                                      )

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

        self.pool = pool

        if monitor is not None:
            self.monitor = monitor
        self.mode = mode
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.cond=cond
        if self.cond:
            self.loss=torch.nn.Identity()

    def calculate_f1_score(self, predictions, target):

        preds = torch.argmax(predictions, dim=1).cpu()
        target=target.cpu()
        f1_micro = f1_score(target.numpy(), preds.numpy(), average='micro')
        f1_macro = f1_score(target.numpy(), preds.numpy(), average='macro')
        return torch.tensor(f1_micro).to(self.device), torch.tensor(f1_macro).to(self.device)

    def calculate_metrics(self, predictions, target):

        # preds = torch.argmax(predictions, dim=1).cpu()

        preds = predictions.cpu().numpy()
        target = target.cpu().numpy()

        roc_auc = roc_auc_score(target, preds)
        pr_auc = average_precision_score(target, preds)

        return torch.tensor(roc_auc).to(self.device), torch.tensor(pr_auc).to(self.device)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, lab_test, time, padding_masks=None):
        """
        Args:
            lab_test: (batch_size, seq_length, feat_dim) torch tensor of lab test data (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        X = lab_test
        b, n, _ = X.shape
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2).float()
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space

        # add cls token
        cls_token = repeat(self.cls_token, '1 1 d -> 1 b d', b=b)
        inp = torch.cat((cls_token, inp), dim=0)

        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in
        # MultiHeadAttention / TransformerEncoderLayer. We prepend a valid CLS mask.
        if padding_masks is not None:
            cls_valid_mask = torch.ones(
                (padding_masks.shape[0], 1),
                dtype=padding_masks.dtype,
                device=padding_masks.device,
            )
            key_padding_mask = torch.cat([cls_valid_mask, padding_masks], dim=1)
            output = self.transformer_encoder(
                inp,
                src_key_padding_mask=~key_padding_mask,
            )  # (seq_length + 1, batch_size, d_model)
        else:
            output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)

        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).

        output = output.mean(dim=1,keepdim=True) if self.pool == 'mean' else output[:, 0]
        ret = self.mlp_head(output)  
        # scores = torch.sigmoid(out)
        return ret.squeeze(dim=1)

    def encode(self, X, padding_masks=None):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            cls: (batch_size, 1, d_model) pooled representation
            output: (batch_size, seq_length, d_model) per-time-step features
        """
        b, n, _ = X.shape
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2).float()
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space

        # # add cls token
        cls_token = repeat(self.cls_token, '1 1 d -> 1 b d', b=b)
        inp = torch.cat((cls_token, inp), dim=0)

        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in
        # MultiHeadAttention / TransformerEncoderLayer. We prepend a valid CLS mask.
        if padding_masks is not None:
            cls_valid_mask = torch.ones(
                (padding_masks.shape[0], 1),
                dtype=padding_masks.dtype,
                device=padding_masks.device,
            )
            key_padding_mask = torch.cat([cls_valid_mask, padding_masks], dim=1)
            output = self.transformer_encoder(
                inp,
                src_key_padding_mask=~key_padding_mask,
            )  # (seq_length + 1, batch_size, d_model)
        else:
            output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)

        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).

        cls = output.mean(dim=1) if self.pool == 'mean' else output[:, 0]

        return cls.unsqueeze(dim=1), output
    
    def encode_per_lab_test(self, X, padding_masks=None):
        """
        Encode lab test time series and extract per-lab-test, per-time-step features.
        
        This method processes the lab test sequence and generates features for each lab test
        at each time step by using lab test values to condition the sequence features.
        
        Args:
            X: (batch_size, seq_length, feat_dim) lab test values where feat_dim = num_lab_tests
            padding_masks: (batch_size, seq_length) boolean tensor
        
        Returns:
            lab_features: (batch_size, seq_length, num_lab_tests, d_model) features for each lab test at each time step
        """
        num_lab_tests = X.shape[-1]
        batch_size, seq_length, _ = X.shape
        
        # Get sequence-level features: (batch_size, seq_length, d_model)
        _, seq_features = self.encode(X, padding_masks)
        # `encode` keeps CLS token in the sequence output; drop it to align with
        # lab time steps (`seq_length`).
        if seq_features.shape[1] == seq_length + 1:
            seq_features = seq_features[:, 1:, :]
        
        # Create projection layer if it doesn't exist
        if not hasattr(self, 'lab_test_projection'):
            self.lab_test_projection = nn.Sequential(
                nn.Linear(self.d_model + 1, self.d_model),  # +1 for lab test value
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model)
            ).to(seq_features.device)
        
        # Expand sequence features and combine with lab test values
        # seq_features: (batch_size, seq_length, d_model)
        # X: (batch_size, seq_length, num_lab_tests)
        
        lab_features_list = []
        for lab_idx in range(num_lab_tests):
            # Get lab test values for this lab: (batch_size, seq_length, 1)
            lab_values = X[:, :, lab_idx:lab_idx+1]
            
            # Combine sequence features with lab test values
            combined = torch.cat([seq_features, lab_values], dim=-1)  # (batch_size, seq_length, d_model+1)
            
            # Project to lab-specific features
            lab_feat = self.lab_test_projection(combined)  # (batch_size, seq_length, d_model)
            lab_features_list.append(lab_feat)
        
        # Stack: (batch_size, seq_length, num_lab_tests, d_model)
        lab_features = torch.stack(lab_features_list, dim=2)
        
        return lab_features

    def get_input(self, batch):
        lab_test = batch[0]
        lab_test = lab_test.to(self.device)

        mask = batch[1]
        mask = mask.to(self.device).to(torch.bool)

        y = batch[2]
        y = y.to(self.device)
        # if self.task!='los':
        #    y=y.float()
        return lab_test, mask, y

    def training_step(self, batch,batch_idx) -> STEP_OUTPUT:
        lab_test, mask, target = self.get_input(batch)

        output = self(lab_test,padding_masks=mask)

        loss = self.loss(output, target)

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=target.shape[0])

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        lab_test, mask, target = self.get_input(batch)

        output = self(lab_test,padding_masks=mask)


        CEloss = self.loss(output, target)

        # if self.task == 'los':
        #     return {'val_loss': CEloss, 'target': target, 'preds': output}

        preds = torch.sigmoid(output)

        # self.log("val/CEloss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
        #          batch_size=target.shape[0])
        return {'val_loss': CEloss, 'target': target, 'preds': preds}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_target = torch.cat([x['target'] for x in outputs])
        all_preds = torch.cat([x['preds'] for x in outputs])

        # if self.task == 'los':
        #     micro_f1, macro_f1 = self.calculate_f1_score(all_preds, all_target)
        #     self.log('val/loss', avg_loss, prog_bar=True, on_epoch=True)
        #     self.log('val/micro_f1', micro_f1, prog_bar=True, on_epoch=True)
        #     self.log('val/macro_f1', macro_f1, prog_bar=True, on_epoch=True)
        #
        # else:
        roc_auc, pr_auc = self.calculate_metrics(all_preds, all_target)
        self.log('val/loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log('val/roc_auc', roc_auc, prog_bar=True, on_epoch=True)
        self.log('val/pr_auc', pr_auc, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        lab_test, mask, target = self.get_input(batch)

#         cls, output = self.encode(lab_test,padding_masks=mask)
        output = self(lab_test, padding_masks=mask)

        # import matplotlib.pyplot as plt
        # # xs, masks, x0, sample_id
        # # print(batch)
        #
        # bs=output.shape[0]
        # # for i in range(bs):
        # #     # print(lab_test[i,:,40:60])
        # #
        # #     array = lab_test[i].cpu().numpy()
        # #     plt.imshow(array, cmap='viridis')
        # #     plt.colorbar()
        # #     plt.show()
        # #
        # #     array=output[i].cpu().numpy()
        # #     plt.imshow(array,cmap='viridis')
        # #     plt.colorbar()
        # #     plt.show()
        # plt.imshow(cls.squeeze().cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.show()
        # torch.save(cls,'mask.pt')
        # exit(0)


        CEloss = self.loss(output, target)

        # if self.task == 'los':
        #     return {'test_loss': CEloss, 'target': target, 'preds': output}

        preds = torch.sigmoid(output)

        # self.log("val/CEloss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
        #          batch_size=target.shape[0])
        return {'test_loss': CEloss, 'target': target, 'preds': preds}

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        all_target = torch.cat([x['target'] for x in outputs])
        all_preds = torch.cat([x['preds'] for x in outputs])

        # if self.task == 'los':
        #     micro_f1, macro_f1 = self.calculate_f1_score(all_preds, all_target)
        #     self.log('test/loss', avg_loss, prog_bar=True, on_epoch=True)
        #     self.log('test/micro_f1', micro_f1, prog_bar=True, on_epoch=True)
        #     self.log('test/macro_f1', macro_f1, prog_bar=True, on_epoch=True)

        # else:
        roc_auc, pr_auc = self.calculate_metrics(all_preds, all_target)
        self.log('test/loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log('test/roc_auc', roc_auc, prog_bar=True, on_epoch=True)
        self.log('test/pr_auc', pr_auc, prog_bar=True, on_epoch=True)


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
#         schedular = CosineAnnealingLR(opt, T_max=self.max_epoches, eta_min=1e-7)
#         return [opt], [schedular]
        return [opt]

