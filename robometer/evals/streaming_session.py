#!/usr/bin/env python3
"""
StreamingSession: server-side stateful session for incremental online reward inference.

Instead of re-processing the full growing frame buffer on every server call, a
StreamingSession caches the ViT patch embeddings for every previously seen frame
and only runs the vision encoder on the **new** frame at each timestep.

This gives a significant speed-up because:
  - ViT encoding of each frame is O(patches) — often the bottleneck for long sequences.
  - The LM still attends to the full sequence (no approximation), so reward quality
    is identical to running the full sequence from scratch each time.

Supported architectures:
  - Qwen2.5-VL in multi-image mode (use_multi_image=True).
    Video mode and SmolVLM require the full pixel_values batch and are NOT currently
    supported for ViT-caching; those model types will fall back to a full re-run.
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional, Tuple

import numpy as np
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.models.utils import convert_bins_to_continuous
from robometer.utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Architecture-agnostic helpers
# ---------------------------------------------------------------------------

def _find_embed_tokens(qwen_vl_model) -> torch.nn.Embedding:
    """Find the token embedding layer across Qwen2.5-VL, Qwen3-VL, and SmolVLM.

    Different versions of the transformers library nest embed_tokens differently:
      - Qwen2.5-VL:  qwen_vl_model.model.embed_tokens
      - Qwen3-VL:    qwen_vl_model.embed_tokens  (no nested .model in some versions)
                     OR qwen_vl_model.model.embed_tokens
      - SmolVLM:     qwen_vl_model.model.embed_tokens  (typically)
    """
    candidates = [
        "model.embed_tokens",
        "embed_tokens",
        "language_model.model.embed_tokens",
        "language_model.embed_tokens",
    ]
    for path in candidates:
        try:
            obj = qwen_vl_model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, torch.nn.Embedding):
                logger.debug(f"Found embed_tokens at {type(qwen_vl_model).__name__}.{path}")
                return obj
        except AttributeError:
            continue

    # Last resort: search all named modules
    for name, module in qwen_vl_model.named_modules():
        if name.endswith("embed_tokens") and isinstance(module, torch.nn.Embedding):
            logger.debug(f"Found embed_tokens via named_modules at: {name}")
            return module

    raise AttributeError(
        f"Could not find embed_tokens in {type(qwen_vl_model).__name__}. "
        f"Top-level attributes: {[n for n, _ in qwen_vl_model.named_children()]}"
    )


def _find_vision_start_token_id(rbm_model) -> Optional[int]:
    """Return the token ID that marks the start of an image span, or None."""
    tokenizer = None
    if hasattr(rbm_model, "processor") and hasattr(rbm_model.processor, "tokenizer"):
        tokenizer = rbm_model.processor.tokenizer
    elif hasattr(rbm_model, "tokenizer"):
        tokenizer = rbm_model.tokenizer
    if tokenizer is None:
        return None
    for tok in ("<|vision_start|>",):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            return int(tid)
    return None


def _find_vision_end_token_id(rbm_model) -> Optional[int]:
    """Return the token ID that marks the end of an image span, or None."""
    tokenizer = None
    if hasattr(rbm_model, "processor") and hasattr(rbm_model.processor, "tokenizer"):
        tokenizer = rbm_model.processor.tokenizer
    elif hasattr(rbm_model, "tokenizer"):
        tokenizer = rbm_model.tokenizer
    if tokenizer is None:
        return None
    for tok in ("<|vision_end|>", "<fake_token_around_image>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            return int(tid)
    return None


def _find_image_token_id(rbm_model) -> int:
    """Find the image-pad token ID across VLM architectures.

    Tries:
      1. rbm_model.model.config.image_token_id   (Qwen2.5-VL standard)
      2. rbm_model.processor.tokenizer (look up "<|image_pad|>")
      3. rbm_model.processor.tokenizer (look up "<image_pad>")
    """
    # Try config attribute
    cfg = getattr(rbm_model.model, "config", None)
    for attr in ("image_token_id", "image_pad_token_id"):
        val = getattr(cfg, attr, None)
        if val is not None:
            logger.debug(f"image_token_id={val} from config.{attr}")
            return int(val)

    # Try processor tokenizer
    tokenizer = None
    if hasattr(rbm_model, "processor") and hasattr(rbm_model.processor, "tokenizer"):
        tokenizer = rbm_model.processor.tokenizer
    elif hasattr(rbm_model, "tokenizer"):
        tokenizer = rbm_model.tokenizer

    if tokenizer is not None:
        for tok in ("<|image_pad|>", "<image_pad>", "<img>"):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                logger.debug(f"image_token_id={tid} from tokenizer token '{tok}'")
                return int(tid)

    raise RuntimeError(
        f"Could not determine image_token_id for {type(rbm_model.model).__name__}. "
        "Set it explicitly or ensure the model config has 'image_token_id'."
    )


class StreamingSession:
    """Stateful per-session inference helper that caches ViT patch embeddings.

    Each session corresponds to one trajectory that is being observed frame-by-frame
    in real time.  On each call to :meth:`append_frame` the session:

    1. Adds the new frame to its internal buffer.
    2. Runs the collator on **all** frames 0..i to get fresh input_ids / attention_mask
       (these change as the sequence grows) and the per-frame pixel_values / grid_thw.
    3. Extracts the **new frame's** pixel_values from the full tensor using grid_thw.
    4. Runs the ViT (vision encoder) **only** on the new frame's patches and caches the
       resulting patch embeddings.
    5. Builds ``inputs_embeds`` by embedding the token sequence and substituting the
       ``<|image_pad|>`` positions with the concatenated cached ViT outputs.
    6. Runs the LM forward with ``pixel_values=None`` so the ViT is skipped entirely.
    7. Extracts the last frame's hidden state and passes it through the progress head.

    Args:
        rbm_model: An ``RBM`` model instance (already on *device*).
        batch_collator: The collator used to tokenise ``ProgressSample`` objects.
        task: Natural-language task description for this trajectory.
        device: Target device (must match where *rbm_model* lives).
        is_discrete_mode: Whether the model uses discrete progress bins.
    """

    def __init__(
        self,
        rbm_model,
        batch_collator,
        task: str,
        device: torch.device | str,
        is_discrete_mode: bool = False,
    ):
        self.rbm = rbm_model
        self.batch_collator = batch_collator
        self.task = task
        self.device = torch.device(device) if isinstance(device, str) else device
        self.is_discrete_mode = is_discrete_mode

        # Growing buffer of raw frames (uint8 numpy arrays, H×W×C).
        self.frames: List[np.ndarray] = []

        # Cached ViT patch embeddings, one tensor per frame.
        # Element j has shape (num_patches_j, hidden_dim).
        self.visual_embeds_cache: List[torch.Tensor] = []

        self.session_id: str = uuid.uuid4().hex
        self.created_at: float = time.time()
        self.last_used: float = time.time()

        # Resolve architecture-specific helpers once at init time
        self._embed_tokens = _find_embed_tokens(rbm_model.model)
        self._image_token_id = _find_image_token_id(rbm_model)
        self._vision_end_token_id = _find_vision_end_token_id(rbm_model)
        self._vision_start_token_id: Optional[int] = _find_vision_start_token_id(rbm_model)

        # LM KV cache state — grows incrementally, one frame at a time.
        # _past_kv: past_key_values from the last forward pass (no suffix).
        # _prev_processed_len: number of tokens already in the KV cache.
        self._past_kv = None
        self._prev_processed_len: int = 0

        # Extra state for evaluate_candidates_batch.
        # _last_ids_no_sfx: suffix-stripped token IDs from the most recent
        #   append_frame call (stored on CPU to save GPU memory).
        # _frame_grid_thws: per-frame (T, H, W) grid stored when each frame
        #   was appended; used to build consistent position_ids.
        self._last_ids_no_sfx: Optional[torch.Tensor] = None   # CPU
        self._frame_grid_thws: List[torch.Tensor] = []          # CPU, each (1, 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_frame(self, frame_np: np.ndarray) -> Tuple[float, float]:
        """Add one frame and return ``(reward, success_prob)`` for that frame.

        Uses two levels of caching to keep each call O(1) in sequence length:

        * **ViT cache**: the vision encoder runs only on the *new* frame's patches.
        * **LM KV cache**: the transformer attention runs only on the *new* frame's
          tokens; all previous token KV pairs are served from ``past_key_values``.

        The suffix appended by the chat template (``<|im_end|>…``) is stripped
        before caching so it does not corrupt the positional encoding of future
        frames.

        Args:
            frame_np: uint8 numpy array of shape ``(H, W, C)``.

        Returns:
            ``(reward, success_prob)`` – float scalars in ``[0, 1]``.
        """
        self.last_used = time.time()
        self.frames.append(frame_np)

        all_frames = np.stack(self.frames)  # (N, H, W, C)
        sample = self._make_progress_sample(all_frames)

        # ---- Tokenise all frames 0..i ----
        batch_out = self.batch_collator([sample])
        prog = batch_out["progress_inputs"]

        input_ids_full: torch.Tensor = prog["input_ids"].to(self.device)      # (1, full_len)
        pixel_values = prog.get("pixel_values")
        image_grid_thw = prog.get("image_grid_thw")

        if pixel_values is None or image_grid_thw is None:
            logger.warning(
                "StreamingSession: pixel_values or image_grid_thw missing – "
                "falling back to full forward pass (no caching)."
            )
            self._past_kv = None
            self._prev_processed_len = 0
            self.visual_embeds_cache.clear()
            return self._full_forward(prog)

        pixel_values = pixel_values.to(self.device)
        image_grid_thw = image_grid_thw.to(self.device)
        model_dtype = self.rbm.model.dtype

        # ---- 1. ViT: only on new frame ----
        thw_cpu = image_grid_thw.cpu()
        num_frames = thw_cpu.shape[0]
        prev_patches = int(sum(
            thw_cpu[j, 0] * thw_cpu[j, 1] * thw_cpu[j, 2]
            for j in range(num_frames - 1)
        ))
        new_n_patches = int(thw_cpu[-1, 0] * thw_cpu[-1, 1] * thw_cpu[-1, 2])
        pv_new = pixel_values[prev_patches: prev_patches + new_n_patches].to(model_dtype)

        with torch.no_grad():
            vit_out = self.rbm.model.visual(pv_new, grid_thw=image_grid_thw[-1:])
            new_embeds = vit_out[0] if isinstance(vit_out, tuple) else vit_out
        self.visual_embeds_cache.append(new_embeds)

        # ---- 2. Strip chat-template suffix so cached positions stay stable ----
        # The collator adds "<|im_end|>…" after the last image; those tokens
        # must not enter the KV cache because they shift position when a new
        # frame is appended next step.
        ids_1d = input_ids_full[0]               # (full_len,)
        ids_no_sfx = self._strip_suffix(ids_1d)  # (L,)  L ≤ full_len
        L = ids_no_sfx.shape[0]

        # Sanity-check: if L < prev_processed_len the collator must have changed
        # the sequence (e.g. truncation) — reset the KV cache.
        if L < self._prev_processed_len:
            logger.warning(
                "StreamingSession: sequence shrank after stripping suffix "
                f"({L} < {self._prev_processed_len}). Resetting KV cache."
            )
            self._past_kv = None
            self._prev_processed_len = 0

        # Attention mask for the full no-suffix sequence (no padding in single-
        # sample inference, so all ones).
        full_attn = torch.ones(1, L, dtype=torch.long, device=self.device)

        # ---- 3. Build inputs_embeds (full no-suffix sequence) ----
        with torch.no_grad():
            inputs_embeds_full = self._embed_tokens(
                ids_no_sfx.unsqueeze(0)
            ).clone()  # (1, L, hidden)

            image_mask = (ids_no_sfx == self._image_token_id)
            all_vis = torch.cat(self.visual_embeds_cache, dim=0).to(model_dtype)

            if image_mask.sum() != all_vis.shape[0]:
                logger.warning(
                    f"StreamingSession: image_mask count ({image_mask.sum()}) "
                    f"!= cached patch count ({all_vis.shape[0]}). Resetting."
                )
                self._past_kv = None
                self._prev_processed_len = 0
                self.visual_embeds_cache.clear()
                return self._full_forward(prog)

            inputs_embeds_full[0, image_mask] = all_vis

        # ---- 4. Compute position_ids for the full no-suffix sequence ----
        qwen_model = self.rbm.model
        position_ids_full: Optional[torch.Tensor] = None
        if hasattr(qwen_model, "get_rope_index"):
            with torch.no_grad():
                try:
                    position_ids_full, _ = qwen_model.get_rope_index(
                        ids_no_sfx.unsqueeze(0),
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=None,
                        attention_mask=full_attn,
                    )
                except TypeError:
                    position_ids_full, _ = qwen_model.get_rope_index(
                        ids_no_sfx.unsqueeze(0),
                        image_grid_thw=image_grid_thw,
                    )

        # ---- 5. Incremental LM forward (KV cached) ----
        prev_len = self._prev_processed_len
        is_first_step = (self._past_kv is None)

        if is_first_step:
            # Process full sequence; warm up the KV cache.
            new_embeds_lm = inputs_embeds_full          # (1, L, hidden)
            pos_ids = position_ids_full                 # (3, 1, L) or None
        else:
            # Only process the tokens added since the last step.
            new_embeds_lm = inputs_embeds_full[:, prev_len:, :]   # (1, new_len, hidden)
            pos_ids = (
                position_ids_full[:, :, prev_len:]                 # (3, 1, new_len)
                if position_ids_full is not None else None
            )

        fw = dict(
            input_ids=None,
            inputs_embeds=new_embeds_lm,
            attention_mask=full_attn,        # full length so model knows total context
            pixel_values=None,               # ViT is skipped
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if pos_ids is not None:
            fw["position_ids"] = pos_ids
        if not is_first_step:
            fw["past_key_values"] = self._past_kv

        with torch.no_grad():
            outputs = qwen_model(**fw)

        # Update KV cache and cursor
        self._past_kv = outputs.past_key_values
        self._prev_processed_len = L

        # Save history state needed by evaluate_candidates_batch.
        # Store on CPU to avoid occupying GPU memory between calls.
        self._last_ids_no_sfx = ids_no_sfx.cpu()
        self._frame_grid_thws.append(image_grid_thw[-1:].cpu())  # (1, 3) for this frame

        # Hidden states: (1, L, hidden) for first step; (1, new_len, hidden) after
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            new_hidden = outputs.last_hidden_state
        else:
            new_hidden = outputs.hidden_states[-1]

        # ---- 6. Extract reward for the current (last) frame ----
        # On the first step the output covers the full sequence; on subsequent
        # steps it covers only the new tokens — both cases handled by taking [-1].
        if is_first_step:
            reward, success_prob = self._extract_reward(new_hidden, ids_no_sfx.unsqueeze(0))
        else:
            new_ids = ids_no_sfx[prev_len:]   # (new_len,)
            reward, success_prob = self._extract_reward(new_hidden, new_ids.unsqueeze(0))
        return reward, success_prob

    def evaluate_candidate_frame(self, frame_np: np.ndarray) -> Tuple[float, float]:
        """Evaluate a candidate frame without advancing the session state.

        Temporarily extends the frame buffer and KV cache to score *frame_np*,
        then restores the session to its pre-call state.  The frame is **not**
        added to the history.  Thread safety: callers must not call
        :meth:`append_frame` concurrently with this method.

        Args:
            frame_np: uint8 numpy array of shape ``(H, W, C)``.

        Returns:
            ``(reward, success_prob)`` — the predicted reward for *frame_np*
            given the current session history, without committing *frame_np*.
        """
        # Snapshot every field that append_frame may mutate.
        # past_key_values tensors are never mutated in-place by the model —
        # append_frame always assigns a fresh object — so saving the reference
        # is sufficient (no deep-copy needed).
        saved_frames = list(self.frames)
        saved_embeds = list(self.visual_embeds_cache)
        saved_past_kv = self._past_kv
        saved_prev_len = self._prev_processed_len
        saved_last_used = self.last_used
        saved_last_ids = self._last_ids_no_sfx
        saved_frame_thws = list(self._frame_grid_thws)

        try:
            result = self.append_frame(frame_np)
        finally:
            # Restore — the newly allocated past_kv tensors are released by GC.
            self.frames = saved_frames
            self.visual_embeds_cache = saved_embeds
            self._past_kv = saved_past_kv
            self._prev_processed_len = saved_prev_len
            self.last_used = saved_last_used
            self._last_ids_no_sfx = saved_last_ids
            self._frame_grid_thws = saved_frame_thws

        return result

    def evaluate_candidates_batch(
        self, frames_np: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Evaluate N candidate frames in **one** batched LM forward.

        Much faster than N sequential :meth:`evaluate_candidate_frame` calls:

        * Each candidate is tokenised **alone** (not merged with history) so
          that Qwen-VL's dynamic per-frame resolution does not corrupt the
          patch-count accounting of cached history embeddings.
        * A synthetic combined token sequence is then constructed by appending
          the candidate's image tokens to ``_last_ids_no_sfx`` (saved by the
          most recent :meth:`append_frame` call).
        * The ViT processes all N new frames in a single batched forward.
        * The LM runs once with the KV cache expanded to batch size N.

        The session state is **not** modified.

        Falls back to sequential :meth:`evaluate_candidate_frame` when the KV
        cache is uninitialised (i.e. no :meth:`append_frame` call has been
        made yet).

        Args:
            frames_np: uint8 array of shape ``(N, H, W, C)``.

        Returns:
            List of ``(reward, success_prob)`` tuples, one per candidate.
        """
        N = frames_np.shape[0]
        if N == 0:
            return []
        if N == 1:
            return [self.evaluate_candidate_frame(frames_np[0])]

        # Need both the KV cache and the saved history ids from append_frame.
        if self._past_kv is None or self._last_ids_no_sfx is None:
            logger.debug(
                "evaluate_candidates_batch: KV cache not initialised, "
                "falling back to sequential evaluation"
            )
            return [self.evaluate_candidate_frame(frames_np[n]) for n in range(N)]

        model_dtype = self.rbm.model.dtype
        qwen_model = self.rbm.model

        # ---- 1. Tokenise each candidate ALONE ----
        # This avoids Qwen-VL dynamic resolution changing history patch counts
        # when the combined sequence length grows beyond a budget threshold.
        all_pv_cand: List[torch.Tensor] = []
        thw_cand: Optional[torch.Tensor] = None   # (1, 3) — same for all N (same res)
        n_cand_patches: Optional[int] = None

        for n in range(N):
            single_sample = self._make_progress_sample(frames_np[n:n+1])
            cand_out = self.batch_collator([single_sample])
            cand_prog = cand_out["progress_inputs"]
            pv = cand_prog.get("pixel_values")
            thw = cand_prog.get("image_grid_thw")
            if pv is None or thw is None:
                logger.warning(
                    "evaluate_candidates_batch: candidate %d missing pixel_values, "
                    "falling back to sequential", n
                )
                return [self.evaluate_candidate_frame(frames_np[k]) for k in range(N)]
            pv = pv.to(self.device).to(model_dtype)
            thw = thw.to(self.device)
            n_patches = int(thw[0, 0] * thw[0, 1] * thw[0, 2])
            if thw_cand is None:
                thw_cand = thw          # (1, 3)
                n_cand_patches = n_patches
            elif n_patches != n_cand_patches:
                # Different patch counts (unusual) — fall back to sequential.
                logger.warning(
                    "evaluate_candidates_batch: candidate %d has %d patches "
                    "(expected %d), falling back to sequential", n, n_patches, n_cand_patches
                )
                return [self.evaluate_candidate_frame(frames_np[k]) for k in range(N)]
            all_pv_cand.append(pv)

        # ---- 2. Batched ViT forward (all N candidate frames at once) ----
        pv_stacked = torch.cat(all_pv_cand, dim=0)           # (N*n_cand_patches_raw, ...)
        thw_repeated = thw_cand.repeat(N, 1)                 # (N, 3)
        with torch.no_grad():
            vit_out = qwen_model.visual(pv_stacked, grid_thw=thw_repeated)
            all_new_embeds = vit_out[0] if isinstance(vit_out, tuple) else vit_out
        # all_new_embeds has shape (N * n_cand_merged, hidden) where n_cand_merged
        # = n_cand_patches_raw / spatial_merge_size².  The merged count is what the
        # tokenizer uses for <|image_pad|> tokens — NOT the raw patch count.
        n_cand_merged = all_new_embeds.shape[0] // N
        embeds_per_candidate = all_new_embeds.split(n_cand_merged, dim=0)
        # embeds_per_candidate: list of N tensors, each (n_cand_merged, hidden)

        # ---- 3. Construct combined token IDs without re-tokenising history ----
        # Append <|vision_start|> image_pad×n_cand_merged <|vision_end|> to saved
        # history ids.  Use the MERGED count so it matches <|image_pad|> usage.
        hist_ids = self._last_ids_no_sfx.to(self.device)   # (L_hist,)
        if (self._vision_start_token_id is None
                or self._vision_end_token_id is None):
            # Cannot construct image tokens without the special token IDs.
            logger.warning(
                "evaluate_candidates_batch: vision start/end token IDs unknown, "
                "falling back to sequential"
            )
            return [self.evaluate_candidate_frame(frames_np[n]) for n in range(N)]

        new_frame_ids = torch.cat([
            torch.tensor([self._vision_start_token_id], device=self.device),
            torch.full((n_cand_merged,), self._image_token_id,
                       dtype=torch.long, device=self.device),
            torch.tensor([self._vision_end_token_id], device=self.device),
        ])                                    # (n_cand_merged + 2,)
        ids_combined = torch.cat([hist_ids, new_frame_ids])   # (L_hist + n_cand_merged + 2,)
        L_combined = ids_combined.shape[0]

        # ---- 4. Build inputs_embeds batch (N, L_combined, hidden) ----
        with torch.no_grad():
            base_embeds = self._embed_tokens(
                ids_combined.unsqueeze(0)
            ).clone()   # (1, L_combined, hidden)

            image_mask = (ids_combined == self._image_token_id)
            hist_vis_count = (
                sum(e.shape[0] for e in self.visual_embeds_cache)
                if self.visual_embeds_cache else 0
            )
            total_img = int(image_mask.sum())
            expected = hist_vis_count + n_cand_merged
            if total_img != expected:
                logger.warning(
                    "evaluate_candidates_batch: unexpected image token count "
                    "after manual concat (%d vs %d), falling back to sequential",
                    total_img, expected
                )
                return [self.evaluate_candidate_frame(frames_np[n]) for n in range(N)]

            img_positions = image_mask.nonzero(as_tuple=True)[0]
            hist_positions = img_positions[:hist_vis_count]
            new_positions  = img_positions[hist_vis_count:]

            # Fill shared history patches (same for all N)
            if self.visual_embeds_cache:
                hist_vis = torch.cat(self.visual_embeds_cache, dim=0).to(model_dtype)
                base_embeds[0, hist_positions] = hist_vis

            # Clone to N and fill per-candidate patches
            inputs_embeds_batch = base_embeds.expand(N, -1, -1).clone()  # (N, L_combined, hidden)
            for n in range(N):
                inputs_embeds_batch[n, new_positions] = embeds_per_candidate[n]

        # ---- 5. Position IDs using saved history grid thws + candidate thw ----
        all_grid_thws: Optional[torch.Tensor] = None
        if self._frame_grid_thws:
            all_grid_thws = torch.cat(
                [g.to(self.device) for g in self._frame_grid_thws] + [thw_cand],
                dim=0,
            )   # (K+1, 3)

        position_ids_full: Optional[torch.Tensor] = None
        full_attn_1 = torch.ones(1, L_combined, dtype=torch.long, device=self.device)
        if hasattr(qwen_model, "get_rope_index") and all_grid_thws is not None:
            with torch.no_grad():
                try:
                    position_ids_full, _ = qwen_model.get_rope_index(
                        ids_combined.unsqueeze(0),
                        image_grid_thw=all_grid_thws,
                        video_grid_thw=None,
                        attention_mask=full_attn_1,
                    )
                except TypeError:
                    position_ids_full, _ = qwen_model.get_rope_index(
                        ids_combined.unsqueeze(0),
                        image_grid_thw=all_grid_thws,
                    )

        # ---- 6. Batched incremental LM forward ----
        prev_len = self._prev_processed_len
        full_attn = torch.ones(N, L_combined, dtype=torch.long, device=self.device)

        new_embeds_lm = inputs_embeds_batch[:, prev_len:, :]   # (N, new_len, hidden)

        if position_ids_full is not None:
            pos_slice = position_ids_full[..., prev_len:]
            if pos_slice.dim() == 3:
                pos_ids = pos_slice.expand(-1, N, -1).contiguous()   # (3, N, new_len)
            else:
                pos_ids = pos_slice.expand(N, -1).contiguous()        # (N, new_len)
        else:
            pos_ids = None

        expanded_past_kv = self._expand_past_kv(self._past_kv, N)

        fw: dict = dict(
            input_ids=None,
            inputs_embeds=new_embeds_lm,
            attention_mask=full_attn,
            pixel_values=None,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if pos_ids is not None:
            fw["position_ids"] = pos_ids
        fw["past_key_values"] = expanded_past_kv

        with torch.no_grad():
            outputs = qwen_model(**fw)

        # ---- 7. Extract per-candidate rewards ----
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            new_hidden = outputs.last_hidden_state   # (N, new_len, hidden)
        else:
            new_hidden = outputs.hidden_states[-1]

        new_ids = new_frame_ids   # (n_cand_merged + 2,) — same structure for all N
        results: List[Tuple[float, float]] = []
        for n in range(N):
            r, s = self._extract_reward(new_hidden[n:n+1], new_ids.unsqueeze(0))
            results.append((r, s))
        return results

    def reset_kv_cache(self) -> None:
        """Discard the LM KV cache (keeps the frame buffer and ViT cache)."""
        self._past_kv = None
        self._prev_processed_len = 0
        self._last_ids_no_sfx = None
        self._frame_grid_thws = []

    @property
    def num_frames(self) -> int:
        """Number of frames appended so far."""
        return len(self.frames)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_past_kv(self, past_kv, N: int):
        """Return *past_kv* with the batch dimension expanded from 1 to *N*.

        Uses :meth:`~torch.Tensor.expand` (zero-copy view) so no data is
        copied.  Handles both ``DynamicCache`` (HuggingFace ≥ 4.38, detected
        via ``isinstance``) and legacy tuple-of-tuples format.
        """
        if past_kv is None:
            return None

        # HuggingFace ≥ 4.38: DynamicCache — detected by isinstance so we
        # don't rely on any internal attribute names that vary across versions.
        # Use the stable to_legacy_cache() / from_legacy_cache() round-trip.
        try:
            from transformers.cache_utils import DynamicCache
            if isinstance(past_kv, DynamicCache):
                legacy = past_kv.to_legacy_cache()   # tuple of (k, v) per layer
                expanded_legacy = tuple(
                    (k.expand(N, -1, -1, -1), v.expand(N, -1, -1, -1))
                    for k, v in legacy
                )
                return DynamicCache.from_legacy_cache(expanded_legacy)
        except (ImportError, AttributeError):
            pass

        # Legacy: plain tuple of (key, value) pairs, one per layer.
        if isinstance(past_kv, tuple):
            return tuple(
                (layer[0].expand(N, -1, -1, -1), layer[1].expand(N, -1, -1, -1))
                for layer in past_kv
            )

        raise TypeError(
            f"Unsupported past_key_values type: {type(past_kv)}. "
            "Expected DynamicCache or tuple of (key, value) pairs."
        )

    def _strip_suffix(self, ids_1d: torch.Tensor) -> torch.Tensor:
        """Return *ids_1d* truncated at (and including) the last vision-end token.

        Everything after the last image span (vision_end token) is the
        chat-template suffix that must not enter the KV cache.
        """
        if self._vision_end_token_id is not None:
            positions = (ids_1d == self._vision_end_token_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                return ids_1d[: positions[-1].item() + 1]
        # Fallback: no stripping (model without separate end token)
        return ids_1d

    def _extract_reward(
        self,
        hidden_state: torch.Tensor,  # (1, seq_len, hidden)  seq_len may be new_len
        input_ids: torch.Tensor,     # (1, seq_len)
    ) -> Tuple[float, float]:
        """Extract progress reward and success prob from the last frame found in *input_ids*."""
        with torch.no_grad():
            frame_embeddings = self.rbm._extract_hidden_states_from_token_pairs(
                hidden_state[0], input_ids[0]
            )  # (num_frames_in_window, hidden_dim)

            if frame_embeddings.shape[0] == 0:
                logger.warning("_extract_reward: no frame embeddings found — returning 0.")
                return 0.0, 0.0

            last_embed = frame_embeddings[-1].unsqueeze(0)  # (1, hidden_dim)

            progress_logit = self.rbm.progress_head(last_embed)
            if self.is_discrete_mode:
                reward = float(
                    convert_bins_to_continuous(
                        progress_logit.detach().cpu().float()
                    ).squeeze()
                )
            else:
                reward = float(torch.sigmoid(progress_logit).detach().cpu().squeeze())

            success_logit = self.rbm.success_head(last_embed)
            success_prob = float(torch.sigmoid(success_logit).detach().cpu().squeeze())

        return reward, success_prob

    def _full_forward(self, prog_inputs: dict) -> Tuple[float, float]:
        """Fallback: run the complete forward pass without ViT caching."""
        input_ids = prog_inputs["input_ids"].to(self.device)
        attention_mask = prog_inputs["attention_mask"].to(self.device)
        pixel_values = prog_inputs.get("pixel_values")
        image_grid_thw = prog_inputs.get("image_grid_thw")

        kw = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values.to(self.device) if pixel_values is not None else None,
            image_grid_thw=image_grid_thw.to(self.device) if image_grid_thw is not None else None,
            return_dict=True,
        )
        with torch.no_grad():
            outputs = self.rbm.model(**kw, output_hidden_states=True)

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs.hidden_states[-1]
        return self._extract_reward(hidden_state, input_ids)

    def _make_progress_sample(self, frames: np.ndarray) -> ProgressSample:
        """Wrap a numpy frame array in a ProgressSample for the collator."""
        T, H, W, C = frames.shape
        traj = Trajectory(
            frames=frames,
            frames_shape=(T, H, W, C),
            task=self.task,
            quality_label="successful",
            data_source="streaming",
            target_progress=None,
            success_label=None,
            predict_last_frame_mask=None,
            metadata={},
        )
        return ProgressSample(trajectory=traj, data_gen_strategy="subsample_task")


class StreamingSessionManager:
    """Thread-safe registry of active :class:`StreamingSession` objects.

    Sessions are stored per-GPU so that cached tensors always live on the
    correct device.

    Args:
        session_timeout_s: Sessions idle for longer than this are evicted on
            the next :meth:`get` or :meth:`create` call.
    """

    def __init__(self, session_timeout_s: float = 1800.0):
        self._sessions: dict[str, StreamingSession] = {}
        self._timeout = session_timeout_s
        import threading
        self._lock = threading.Lock()

    def create(
        self,
        rbm_model,
        batch_collator,
        task: str,
        device: torch.device | str,
        is_discrete_mode: bool = False,
    ) -> StreamingSession:
        """Create a new session and register it.  Returns the session."""
        session = StreamingSession(
            rbm_model=rbm_model,
            batch_collator=batch_collator,
            task=task,
            device=device,
            is_discrete_mode=is_discrete_mode,
        )
        with self._lock:
            self._evict_stale()
            self._sessions[session.session_id] = session
        logger.info(f"StreamingSession created: id={session.session_id} task='{task}'")
        return session

    def get(self, session_id: str) -> StreamingSession:
        """Return the session for *session_id* or raise :exc:`KeyError`."""
        with self._lock:
            self._evict_stale()
            session = self._sessions[session_id]  # raises KeyError if absent
        return session

    def delete(self, session_id: str) -> bool:
        """Remove and destroy a session.  Returns ``True`` if it existed."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            logger.info(f"StreamingSession deleted: id={session_id}")
            return True
        return False

    def _evict_stale(self) -> None:
        """Remove sessions that have been idle for longer than *timeout_s*."""
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s.last_used > self._timeout
        ]
        for sid in stale:
            logger.info(f"StreamingSession evicted (idle timeout): id={sid}")
            del self._sessions[sid]

    def active_count(self) -> int:
        """Number of currently registered sessions."""
        with self._lock:
            return len(self._sessions)
