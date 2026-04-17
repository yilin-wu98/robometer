#!/usr/bin/env python3
"""
Streaming eval server for incremental / online reward inference.

Adds three new endpoints on top of the standard eval_server endpoints:

  POST /stream/create
      JSON body: {"task": "Pick up the red block"}
      Returns:   {"session_id": "<hex>", "gpu_id": 0}

  POST /stream/{session_id}/append
      Multipart body: one field named "frame" whose value is a .npy binary blob
                      containing a uint8 array of shape (H, W, C).
      Returns: {"session_id": "...", "num_frames": 5,
                "reward": 0.73, "success_prob": 0.21,
                "elapsed_s": 0.12}

  DELETE /stream/{session_id}
      Returns: {"deleted": true, "session_id": "..."}

Usage (start server):
  uv run python robometer/evals/eval_server_streaming.py \\
      model_path=robometer/Robometer-4B \\
      num_gpus=1 \\
      server_port=8001

You can also import ``create_streaming_app`` and mount it into an existing app.
"""

from __future__ import annotations

import copy
import io
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from hydra import main as hydra_main
from omegaconf import DictConfig

from robometer.configs.eval_configs import EvalServerConfig
from robometer.configs.experiment_configs import ExperimentConfig
from robometer.evals.eval_server import (
    MultiGPUEvalServer,
    create_app as create_base_app,
)
from robometer.evals.streaming_session import StreamingSession, StreamingSessionManager
from robometer.utils.config_utils import convert_hydra_to_dataclass, display_config
from robometer.utils.logger import get_logger, setup_loguru_logging
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator, setup_model_and_processor

LOG_LEVEL = "INFO"
setup_loguru_logging(log_level=LOG_LEVEL)
logger = get_logger()


# ---------------------------------------------------------------------------
# Helper: read a .npy blob from raw bytes
# ---------------------------------------------------------------------------

def _load_npy_bytes(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    return np.load(buf, allow_pickle=False)


# ---------------------------------------------------------------------------
# Streaming extension
# ---------------------------------------------------------------------------

def add_streaming_routes(
    app: FastAPI,
    multi_gpu_server: MultiGPUEvalServer,
    session_timeout_s: float = 1800.0,
) -> StreamingSessionManager:
    """Attach streaming endpoints to an existing FastAPI *app*.

    Each session is pinned to a single GPU (the one with the lowest index that
    is currently idle when the session is created).  After creation the session
    always uses the same GPU replica, so no cross-device tensor copies occur.

    Returns the :class:`StreamingSessionManager` (useful for testing / admin).
    """
    manager = StreamingSessionManager(session_timeout_s=session_timeout_s)

    # GPU-0 replica is used for all streaming sessions.
    # For multi-GPU deployments you could shard sessions across GPUs; the simple
    # approach below always uses GPU-0 which avoids pool contention.
    gpu_info_streaming: Dict[str, Any] = {}

    def _get_gpu_info() -> Dict[str, Any]:
        """Return a cached reference to the GPU-0 model replica.
        The replica is borrowed (not taken from the pool) so that streaming
        sessions do not block batch inference.
        """
        if not gpu_info_streaming:
            # Borrow GPU-0 items once and keep a permanent reference.
            # We deep-copy to avoid interfering with the pool replica.
            gpu0: Dict[str, Any] = None
            # Temporarily pull from the queue to read GPU-0 info, then put back.
            tmp = multi_gpu_server.gpu_pool.get(timeout=30)
            multi_gpu_server.gpu_pool.put(tmp)
            # Deep-copy the model and collator for exclusive streaming use.
            gpu_info_streaming["model"] = copy.deepcopy(tmp["model"])
            gpu_info_streaming["model"].eval()
            gpu_info_streaming["batch_collator"] = copy.deepcopy(tmp["batch_collator"])
            gpu_info_streaming["device"] = tmp["device"]
            gpu_info_streaming["gpu_id"] = tmp["gpu_id"]
            logger.info(
                f"Streaming replica initialised on {tmp['device']} "
                f"(GPU {tmp['gpu_id']})"
            )
        return gpu_info_streaming

    def _is_discrete() -> bool:
        progress_loss_type = getattr(
            multi_gpu_server.exp_config.loss, "progress_loss_type", "l2"
        )
        return progress_loss_type.lower() == "discrete"

    # ------------------------------------------------------------------ #
    #  POST /stream/create                                                 #
    # ------------------------------------------------------------------ #
    @app.post("/stream/create")
    async def stream_create(body: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new streaming session for online reward inference.

        Body JSON fields:
          - ``task`` (str, required): task description.
        """
        task = body.get("task", "")
        if not isinstance(task, str):
            raise HTTPException(status_code=422, detail="'task' must be a string")

        import asyncio
        loop = asyncio.get_event_loop()

        def _create():
            info = _get_gpu_info()
            session = manager.create(
                rbm_model=info["model"],
                batch_collator=info["batch_collator"],
                task=task,
                device=info["device"],
                is_discrete_mode=_is_discrete(),
            )
            return session, info["gpu_id"]

        session, gpu_id = await loop.run_in_executor(None, _create)
        logger.info(f"[stream] Created session {session.session_id} task='{task}'")
        return {"session_id": session.session_id, "gpu_id": gpu_id}

    # ------------------------------------------------------------------ #
    #  POST /stream/{session_id}/append                                    #
    # ------------------------------------------------------------------ #
    @app.post("/stream/{session_id}/append")
    async def stream_append(session_id: str, request: Request) -> Dict[str, Any]:
        """Append one frame and return the reward for the current timestep.

        Multipart body:
          - ``frame``: .npy binary blob with a uint8 array of shape (H, W, C).
        """
        # Read multipart form
        form = await request.form()
        frame_field = form.get("frame")
        if frame_field is None:
            raise HTTPException(status_code=422, detail="Multipart field 'frame' is required")

        frame_bytes: bytes = await frame_field.read()
        frame_np = _load_npy_bytes(frame_bytes)

        if frame_np.ndim != 3:
            raise HTTPException(
                status_code=422,
                detail=f"Expected frame of shape (H, W, C), got shape {frame_np.shape}"
            )
        if frame_np.dtype != np.uint8:
            # Accept float frames but convert to uint8
            if np.issubdtype(frame_np.dtype, np.floating):
                frame_np = (np.clip(frame_np, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

        # Retrieve session
        try:
            session: StreamingSession = manager.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        import asyncio
        loop = asyncio.get_event_loop()

        t0 = time.time()

        def _append():
            return session.append_frame(frame_np)

        reward, success_prob = await loop.run_in_executor(None, _append)
        elapsed = time.time() - t0

        logger.debug(
            f"[stream] session={session_id} frame={session.num_frames} "
            f"reward={reward:.4f} success={success_prob:.4f} elapsed={elapsed:.3f}s"
        )
        return {
            "session_id": session_id,
            "num_frames": session.num_frames,
            "reward": reward,
            "success_prob": success_prob,
            "elapsed_s": round(elapsed, 4),
        }

    # ------------------------------------------------------------------ #
    #  DELETE /stream/{session_id}                                         #
    # ------------------------------------------------------------------ #
    @app.delete("/stream/{session_id}")
    async def stream_delete(session_id: str) -> Dict[str, Any]:
        """Delete a streaming session and free its cached state."""
        deleted = manager.delete(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return {"deleted": True, "session_id": session_id}

    # ------------------------------------------------------------------ #
    #  POST /stream/{session_id}/evaluate_candidates  (batched)            #
    # ------------------------------------------------------------------ #
    @app.post("/stream/{session_id}/evaluate_candidates")
    async def stream_evaluate_candidates(session_id: str, request: Request) -> Dict[str, Any]:
        """Evaluate N candidate frames in one batched LM forward.

        Sends all N frames through the ViT in a single batched call, then runs
        the LM once with the KV cache expanded to batch size N.  Much faster
        than N sequential ``/evaluate_candidate`` calls when N is large.

        The session history is **not** modified.

        Multipart body:
          - ``frames``: .npy binary blob with a uint8 array of shape (N, H, W, C).
        """
        form = await request.form()
        frames_field = form.get("frames")
        if frames_field is None:
            raise HTTPException(status_code=422, detail="Multipart field 'frames' is required")

        frames_bytes: bytes = await frames_field.read()
        frames_np = _load_npy_bytes(frames_bytes)

        if frames_np.ndim != 4:
            raise HTTPException(
                status_code=422,
                detail=f"Expected frames of shape (N, H, W, C), got {frames_np.shape}"
            )
        if frames_np.dtype != np.uint8:
            if np.issubdtype(frames_np.dtype, np.floating):
                frames_np = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                frames_np = frames_np.astype(np.uint8)

        try:
            session: StreamingSession = manager.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        import asyncio
        loop = asyncio.get_event_loop()

        t0 = time.time()

        def _evaluate_batch():
            return session.evaluate_candidates_batch(frames_np)

        results = await loop.run_in_executor(None, _evaluate_batch)
        elapsed = time.time() - t0

        rewards = [r for r, s in results]
        success_probs = [s for r, s in results]

        logger.debug(
            f"[stream] evaluate_candidates session={session_id} N={len(results)} "
            f"rewards={[f'{r:.3f}' for r in rewards]} elapsed={elapsed:.3f}s"
        )
        return {
            "session_id": session_id,
            "num_candidates": len(results),
            "rewards": rewards,
            "success_probs": success_probs,
            "elapsed_s": round(elapsed, 4),
        }

    # ------------------------------------------------------------------ #
    #  POST /stream/{session_id}/evaluate_candidate                        #
    # ------------------------------------------------------------------ #
    @app.post("/stream/{session_id}/evaluate_candidate")
    async def stream_evaluate_candidate(session_id: str, request: Request) -> Dict[str, Any]:
        """Score a candidate frame WITHOUT adding it to the session history.

        Temporarily extends the session's KV cache to evaluate *frame*, then
        restores all state.  Use this to rank N predicted frames at each policy
        query step without corrupting the rolling history.

        Multipart body:
          - ``frame``: .npy binary blob with a uint8 array of shape (H, W, C).
        """
        form = await request.form()
        frame_field = form.get("frame")
        if frame_field is None:
            raise HTTPException(status_code=422, detail="Multipart field 'frame' is required")

        frame_bytes: bytes = await frame_field.read()
        frame_np = _load_npy_bytes(frame_bytes)

        if frame_np.ndim != 3:
            raise HTTPException(
                status_code=422,
                detail=f"Expected frame of shape (H, W, C), got shape {frame_np.shape}"
            )
        if frame_np.dtype != np.uint8:
            if np.issubdtype(frame_np.dtype, np.floating):
                frame_np = (np.clip(frame_np, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

        try:
            session: StreamingSession = manager.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        import asyncio
        loop = asyncio.get_event_loop()

        t0 = time.time()

        def _evaluate():
            return session.evaluate_candidate_frame(frame_np)

        reward, success_prob = await loop.run_in_executor(None, _evaluate)
        elapsed = time.time() - t0

        logger.debug(
            f"[stream] evaluate_candidate session={session_id} "
            f"reward={reward:.4f} success={success_prob:.4f} elapsed={elapsed:.3f}s"
        )
        return {
            "session_id": session_id,
            "reward": reward,
            "success_prob": success_prob,
            "elapsed_s": round(elapsed, 4),
        }

    # ------------------------------------------------------------------ #
    #  GET /stream/status                                                  #
    # ------------------------------------------------------------------ #
    @app.get("/stream/status")
    async def stream_status() -> Dict[str, Any]:
        """Return how many streaming sessions are currently active."""
        return {"active_sessions": manager.active_count()}

    return manager


def create_streaming_app(
    cfg: EvalServerConfig,
    multi_gpu_server: Optional[MultiGPUEvalServer] = None,
    session_timeout_s: float = 1800.0,
) -> FastAPI:
    """Create a FastAPI app that includes both batch and streaming endpoints."""
    if multi_gpu_server is None:
        multi_gpu_server = MultiGPUEvalServer(
            model_path=cfg.model_path,
            num_gpus=getattr(cfg, "num_gpus", None),
            max_workers=getattr(cfg, "max_workers", None),
        )

    # Build the base app (batch endpoints) with the explicit server instance
    app = create_base_app(cfg, multi_gpu_server)

    # Attach streaming endpoints, sharing the same model replicas
    add_streaming_routes(app, multi_gpu_server, session_timeout_s=session_timeout_s)
    logger.info(
        "Streaming endpoints registered: /stream/create, /stream/{id}/append, "
        "/stream/{id}/evaluate_candidates, /stream/{id}/evaluate_candidate, "
        "DELETE /stream/{id}"
    )
    return app


# ---------------------------------------------------------------------------
# Entry point (Hydra)
# ---------------------------------------------------------------------------

@hydra_main(version_base=None, config_path="../configs", config_name="eval_config_server")
def main(cfg: DictConfig):
    eval_cfg = convert_hydra_to_dataclass(cfg, EvalServerConfig)
    display_config(eval_cfg)

    if not eval_cfg.model_path:
        raise ValueError("Eval config must set model_path to a pretrained checkpoint.")

    multi_gpu_server = MultiGPUEvalServer(
        model_path=eval_cfg.model_path,
        num_gpus=eval_cfg.num_gpus,
        max_workers=eval_cfg.max_workers,
    )

    app = create_streaming_app(eval_cfg, multi_gpu_server)
    print(f"Running streaming eval server on {eval_cfg.server_url}:{eval_cfg.server_port}")
    uvicorn.run(app, host=eval_cfg.server_url, port=eval_cfg.server_port)


if __name__ == "__main__":
    main()
