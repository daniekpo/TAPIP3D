from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm
from einops import repeat

from ...datasets.providers.base_provider import BaseDataProvider
from ...datasets.datatypes import RawSliceData, SliceData
from PIL import Image
import cv2
import logging

from ...datasets.utils.dataclass_utils import load_dataclass
from dataclasses import dataclass

from ...datasets.utils.crop_utils import get_crop_args
from evaluation.tapvid3d_splits import MINIVAL_FILES, FULL_EVAL_FILES

logger = logging.getLogger(__name__)


class TAPVid3dProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, override_anno: Optional[str] = None):
        super().__init__(cfg, override_anno)
        self.split: str = cfg.split
        self.data_root: str = cfg.data_root
        self.subset: str = cfg.subset
        self.split: str = cfg.split
        self.only_first_n_seqs: Optional[int] = cfg.get("only_first_n_seqs", None)

        assert self.subset in ["drivetrack", "adt", "pstudio"], f"subset must be either 'drivetrack' or 'adt' or 'pstudio', got {self.subset}"
        assert self.split in ["full", "minival", "sample"], f"split must be either 'full' or 'minival' or 'sample', got {self.split}"

        if self.split == "sample":
            self.npzs = [file for file in os.listdir(os.path.join(self.data_root, self.subset)) if file.endswith('.npz')]
        else:
            self.npzs = MINIVAL_FILES[self.subset] if self.split == 'minival' else FULL_EVAL_FILES[self.subset]

        assert all(os.path.exists(os.path.join(self.data_root, self.subset, f)) for f in self.npzs), "All npz files must exist"
        self.npzs.sort()

        logger.info(f"Loaded TAP-Vid3D/{self.subset} ({self.split}) from {self.data_root} with {len(self.npzs)} sequences")

    def _decode_jpeg(self, data: Any):
        byte_array = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _decode_depth_png(self, data: Any):
        byte_array = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(byte_array, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 1000.0
        return img

    def _resize_rgb(self, img: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(img, resolution[::-1], interpolation=cv2.INTER_LANCZOS4)

    def _resize_depth(self, img: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(img, resolution[::-1], interpolation=cv2.INTER_NEAREST_EXACT)
    
    def _load_slice(
        self, 
        seq_id: int, 
        start: int, 
        length: int, 
        stride: int,
        rng: np.random.Generator,
        executor: ThreadPoolExecutor,
    ) -> RawSliceData:
        npz_path = os.path.join(self.data_root, self.subset, self.npzs[seq_id])

        with open(npz_path, 'rb') as in_f:
            in_npz = np.load(in_f, allow_pickle=True)
            images_jpeg_bytes = in_npz['images_jpeg_bytes'][start:start+length*stride:stride]
            if 'depths_png_bytes' in in_npz:
                depths_png_bytes = in_npz['depths_png_bytes'][start:start+length*stride:stride]
            else:
                depths_png_bytes = None
            queries_xyt = in_npz['queries_xyt']
            tracks_xyz = in_npz['tracks_XYZ'][start:start+length*stride:stride]
            visibles = in_npz['visibility'][start:start+length*stride:stride]
            intrinsics_params = in_npz['fx_fy_cx_cy'][start:start+length*stride:stride]
            in_npz.close()
        
        rgbs, depths = [], []
        rgb_futures, depth_futures = [], []
        for i in range(len(images_jpeg_bytes)):
            rgb_futures.append(executor.submit(self._decode_jpeg, images_jpeg_bytes[i]))
            if depths_png_bytes is not None:
                depth_futures.append(executor.submit(self._decode_depth_png, depths_png_bytes[i]))
        rgbs = [f.result() for f in rgb_futures]
        if depths_png_bytes is not None:
            depths = [f.result() for f in depth_futures]
        else:
            depths = []

        assert all(rgb.shape == rgbs[0].shape for rgb in rgbs), "All images must have the same shape"
        orig_h, orig_w = rgbs[0].shape[:2]
        fx, fy, cx, cy = intrinsics_params

        intrinsics = repeat(
            np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32),
            'i j -> t i j', t=len(rgbs),
        )
        
        queries_t = queries_xyt[..., -1].astype(np.int32)

        T = len(rgbs)
        N = queries_t.shape[0]
        queries_xyz = tracks_xyz[queries_t, np.arange(N)]
        queries = np.concatenate([queries_t[..., None], queries_xyz], axis=-1).astype(np.float32)

        if len(depths) > 0:
            depths = np.stack(depths)
        else:
            depths = np.ones((T, orig_h, orig_w), dtype=np.float32) # should never be used in later code

        # with trajs_3d specified in global space, we cannot set extrinsics arbitrarily
        dummy_extrinsics = repeat(np.eye(4, dtype=np.float32), 'i j -> t i j', t=T)

        rgbs = np.stack(rgbs)
        
        randperm = np.random.permutation(N)
        return RawSliceData.create(
            seq_name=self.npzs[seq_id],
            seq_id=seq_id,
            gt_trajs_3d=tracks_xyz.astype(np.float32),
            gt_query_point=queries,
            visibs=visibles,
            valids=np.ones_like(visibles, dtype=np.bool_), 
            rgbs=rgbs,
            gt_depths=depths,
            gt_intrinsics=intrinsics,
            gt_extrinsics=dummy_extrinsics,
            orig_resolution=np.array([rgbs[0].shape[0], rgbs[0].shape[1]], dtype=np.int32), # (h, w)
            copy_gt_to_est=True,
        )
    
    def load_seq_lens(self) -> List[int]:
        seq_lens = []
        for npz in tqdm(self.npzs, "Loading Sequence Lengths"):
            npz_path = os.path.join(self.data_root, self.subset, npz)
            with open(npz_path, 'rb') as in_f:
                try:
                    in_npz = np.load(in_f, allow_pickle=True)
                except:
                    print(f"Error loading {npz_path}")
                seq_lens.append(len(in_npz['visibility']))
                in_npz.close()
        return seq_lens
