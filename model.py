import torch
import torch.nn as nn

from mamba_ssm.models.mixer_seq_simple import MixerModel


class BaseModel(nn.Module):
    def __init__(
        self,
        n_dims=20,
        n_embd=128,
        n_layer=12,
        interleave=True,
        vocab_size=-1,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.interleave = interleave
        self.tokenized = (vocab_size > 0)
        assert not (self.interleave and self.tokenized)

        self.n_dims = -1 if self.tokenized else n_dims
        self.vocab_size = vocab_size if vocab_size > 0 else 50257
        print(f"Interleaving samples: {interleave}. Tokenized: {self.tokenized}. Vocab size: {vocab_size}.")

        self._read_in = nn.Embedding(vocab_size, n_embd) if self.tokenized else nn.Linear(n_dims, n_embd)
        self._backbone = None
        if self.tokenized:
            self._read_out = nn.Linear(n_embd, vocab_size, bias=False)
            self.tie_weights()
        elif self.interleave:
            self._read_out = nn.Linear(n_embd, 1)
        else:
            self._read_out = nn.Linear(n_embd, n_dims)

    def _combine(self, xs_b, ys_b):
        """
        Interleaves the x's and the y's into a single sequence.

        Returns:
        zs: Input to _read_in linear layer. Shape depends on tokenization.
            (bsize, 2*points) if tokenized into input_ids
            (bsize, 2 * points, dim) if points are R^dim vectors
        """
        if self.tokenized:
            (bsize, points), dim = xs_b.shape, 1
        else:
            bsize, points, dim = xs_b.shape
        zs = torch.stack((xs_b, ys_b), dim=2)
        zs = zs.view(bsize, 2 * points) if self.tokenized else zs.view(bsize, 2 * points, dim)
        return zs

    def tie_weights(self):
        """Ties projections in and out to be the same."""
        self._read_out.weight = self._read_in.weight

    def compile(self, xs, ys, inds):
        """Determines which points to predict for. Then combines xs, ys to zs."""
        if inds is None and self.interleave:
            inds = torch.arange(ys.shape[1])
        elif not self.interleave:  # Only predict on last vector
            inds = -1
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        zs = self._combine(xs, ys) if self.interleave else xs
        return zs, inds

    def masked_predictions(self, prediction, inds):
        """
        Filters the final prediction into the predictions that matter.

        Parameters:
        prediction: Final predictions for all points.
        inds: Indices of which predictions matter for the objective; e.g., for interleave, this is all even indices.
        """
        # Filter by inds
        if self.interleave:
            return prediction[:, ::2, 0][:, inds]  # predict only on xs
        return prediction[:, inds, :]
    
    def forward(self):
        raise NotImplementedError("The forward method must be implemented by the subclass")
    

class MambaMixerModel(BaseModel):
    def __init__(
        self,
        n_dims=20,
        n_embd=128,
        n_layer=12,
        interleave=True,
        vocab_size=-1,
        s4=False,
        mamba_style_block=True,
        mixed_attn=None,
        n_positions=-1,
    ):
        super().__init__(
            n_dims=n_dims,
            n_embd=n_embd,
            n_layer=n_layer,
            interleave=interleave,
            vocab_size=vocab_size,
        )
        self.mixed_attn = mixed_attn
        if mixed_attn == "standard":
            assert n_positions > 0
            self.wpe = nn.Embedding(n_positions, n_embd)

        self.name = f"{'s4' if s4 else 'mamba'}_embd={n_embd}_layer={n_layer}" 
        self._backbone = MixerModel(
            d_model=n_embd,
            n_layer=n_layer,
            s4=s4,
            mamba_style_block=mamba_style_block,
            mixed_attn=mixed_attn,
            block_size=n_positions,
            vocab_size=1, # unused
        )


    def forward(self, input_ids, labels=None):
        """
        Forward pass for causal language modeling.

        Parameters:
        input_ids: Tensor of input token IDs.
        labels: Tensor of target token IDs for computing loss (optional).

        Returns:
        If labels are provided, returns the loss. Otherwise, returns logits.
        """
        embeds = self._read_in(input_ids)
        if self.mixed_attn == "standard":
            pos = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)
            embeds += pos_emb
        output = self._backbone(input_ids=None, inputs_embeds=embeds)
        logits = self._read_out(output)

        if labels is not None:
            # Shift logits and labels for causal language modeling
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss

        return logits