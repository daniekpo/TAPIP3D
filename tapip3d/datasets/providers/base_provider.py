from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from omegaconf import DictConfig
from ...datasets.datatypes import RawSliceData
from pathlib import Path
from ...annotation.utils import load_annotations
from hydra import compose, initialize_config_dir
from einops import repeat
from hydra.core.global_hydra import GlobalHydra

ANNO_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "annotation"

def distance_to_depth(coords: np.ndarray, distances: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    # coords: (..., 2), distances: (...), intrinsics: (..., 3, 3)
  
    coords_homogeneous = np.concatenate((coords, np.ones_like(coords[..., :1])), axis=-1).astype(np.float32)
    K_inv = np.linalg.inv(intrinsics)
    camera_coords = np.einsum("...ij, ...j -> ...i", K_inv, coords_homogeneous)
    depth = camera_coords[..., -1] * distances / np.sqrt((camera_coords ** 2).sum(-1))

    return depth

class BaseDataProvider(ABC):
    def __init__(self, cfg: DictConfig, override_anno: Optional[str]):
        self.cfg = cfg
        self.override_anno: Optional[str] = override_anno
        if self.override_anno is not None:
            GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=str(ANNO_CONFIG_PATH)):
                self.anno_config = compose(config_name=self.override_anno)
            dirs = override_anno.split('/') # type: ignore
            for dir in dirs[:-1]:
                self.anno_config = self.anno_config[dir]
        else:
            self.anno_config = None
            
    def _world_to_local(self, world_coords, intrinsics, extrinsics):
        world_coords_homo = np.concatenate([world_coords, np.ones_like(world_coords[..., :1])], axis=-1)
        local_coords = np.einsum("...ij,...j->...i", extrinsics, world_coords_homo)[..., :3]
        local_coords_2d = np.einsum("...ij,...j->...i", intrinsics, local_coords)[..., :2]
        local_coords_2d /= local_coords[..., 2:3]
        return np.concatenate([local_coords_2d, local_coords[..., -1:]], axis=-1)
    
    def _local_to_world(self, local_coords, intrinsics, extrinsics):
        inv_intrinsics = np.linalg.inv(intrinsics)
        inv_extrinsics = np.linalg.inv(extrinsics)
        local_coords_homo = np.concatenate([local_coords[..., :2], np.ones_like(local_coords[..., :1])], axis=-1)
        camera_coords = np.einsum("...ij,...j->...i", inv_intrinsics, local_coords_homo)[..., :3]
        camera_coords = camera_coords * local_coords[..., 2:3] / camera_coords[..., 2:3]
        camera_coords_homo = np.concatenate([camera_coords, np.ones_like(camera_coords[..., :1])], axis=-1)
        world_coords = np.einsum("...ij,...j->...i", inv_extrinsics, camera_coords_homo)[..., :3]
        return world_coords

    def load_slice(
        self, 
        seq_id: int, 
        start: int, 
        length: int, 
        stride: int,
        rng: np.random.Generator,
        executor: ThreadPoolExecutor,
    ) -> RawSliceData:
        raw_slice_data = self._load_slice(seq_id=seq_id, start=start, length=length, stride=stride, rng=rng, executor=executor)
        
        if self.anno_config is not None:
            assert not 'query_point' in self.anno_config and not 'trajs_3d' in self.anno_config, "query_point and trajs_3d must not be overwritten!"

            raw_slice_data.annotated = True
            raw_slice_data.same_scale = False

            for key in self.anno_config.overrides:
                if key != 'depths':
                    assert f"est_{key}" in raw_slice_data.__dict__, f"Invalid key {key} in annotation!"

                    # we need to convert query to query2d
                    if self.anno_config.overrides[key].lower() == "dummy_extrinsics": # type: ignore
                        T = raw_slice_data.rgbs.shape[0]
                        setattr(raw_slice_data, f"est_{key}", repeat(np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32), 'i j -> t i j', t=T)) # type: ignore
                    else:
                        setattr(raw_slice_data, f"est_{key}", load_annotations(annotation_path=self.anno_config.overrides[key], key=key, seq_id=seq_id, start=start, length=length, stride=stride).astype(np.float32)) # type: ignore
                else:
                    if self.anno_config.overrides[key].lower() == "dummy_depths": # type: ignore
                        raw_slice_data.est_depths = np.ones_like(raw_slice_data.gt_depths)
                    elif self.anno_config.overrides[key].lower() == "copy_gt_depths": # type: ignore
                        raw_slice_data.est_depths = raw_slice_data.gt_depths.copy()
                    elif self.anno_config.overrides[key].lower().startswith("mul"):
                        mul_factor = float(self.anno_config.overrides[key].split(":")[0].lower()[3:])
                        raw_slice_data.est_depths = load_annotations(annotation_path=self.anno_config.overrides[key].split(":")[1], key=key, seq_id=seq_id, start=start, length=length, stride=stride).astype(np.float32) * mul_factor
                    else:
                        raw_slice_data.est_depths = load_annotations(annotation_path=self.anno_config.overrides[key], key=key, seq_id=seq_id, start=start, length=length, stride=stride).astype(np.float32) # type: ignore

                    if raw_slice_data.est_depths.shape != raw_slice_data.rgbs.shape[:3]:
                        print ("Debug: resizing depth with bilinear")
                        futures = []
                        for t in range(raw_slice_data.est_depths.shape[0]):
                            futures.append(executor.submit(lambda img: cv2.resize(img, (raw_slice_data.rgbs.shape[2], raw_slice_data.rgbs.shape[1]), interpolation=cv2.INTER_LINEAR), raw_slice_data.est_depths[t]))
                        raw_slice_data.est_depths = np.stack([future.result() for future in futures])
            trajs_2d = self._world_to_local(raw_slice_data.gt_trajs_3d, raw_slice_data.gt_intrinsics[:, None], raw_slice_data.gt_extrinsics[:, None])[..., :2]

            trajs_2d_clipped = trajs_2d.copy()
            trajs_2d_clipped[..., 0] = np.clip(trajs_2d_clipped[..., 0], 0, raw_slice_data.rgbs.shape[2] - 1)
            trajs_2d_clipped[..., 1] = np.clip(trajs_2d_clipped[..., 1], 0, raw_slice_data.rgbs.shape[1] - 1)
            trajs_2d_int = np.round(trajs_2d_clipped).astype(np.int32)

            T, N = trajs_2d_int.shape[0], trajs_2d_int.shape[1]
            
            filled_imgs = []
            for t in range(T):
                if (raw_slice_data.est_depths[t] == 0).any():
                    filled_img = raw_slice_data.est_depths[t].copy()
                    mask = (filled_img == 0)
                    distance, labels = cv2.distanceTransformWithLabels(mask.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                    labels_to_depth = np.zeros(labels.max() + 1, dtype=np.float32)
                    labels_to_depth[labels[~mask]] = filled_img[~mask]
                    filled_img[mask] = labels_to_depth[labels[mask]]
                    filled_imgs.append(filled_img)
                else:
                    filled_imgs.append(raw_slice_data.est_depths[t].copy())

            filled_imgs = np.stack(filled_imgs)

            trajs_2d_depths = filled_imgs[repeat(np.arange(T), 't -> (t n)', n=N), trajs_2d_int[..., 1].reshape((T*N, )), trajs_2d_int[..., 0].reshape((T*N,))].reshape(T, N)

            trajs_local = np.concatenate((trajs_2d, trajs_2d_depths[..., None]), axis=-1)
            raw_slice_data.est_trajs_3d = self._local_to_world(trajs_local, raw_slice_data.est_intrinsics[:, None], raw_slice_data.est_extrinsics[:, None])
            
            if raw_slice_data.gt_query_point is not None:
                query_frames = raw_slice_data.gt_query_point[..., 0].astype(np.int32)
                query_point_2d = self._world_to_local(raw_slice_data.gt_query_point[..., 1:], raw_slice_data.gt_intrinsics[query_frames], raw_slice_data.gt_extrinsics[query_frames])[..., :2]
                query_point_2d_int = np.round(query_point_2d).astype(np.int32)
                query_point_2d_int[..., 0] = np.clip(query_point_2d_int[..., 0], 0, raw_slice_data.rgbs.shape[2] - 1)
                query_point_2d_int[..., 1] = np.clip(query_point_2d_int[..., 1], 0, raw_slice_data.rgbs.shape[1] - 1)
                query_depths = filled_imgs[query_frames, query_point_2d_int[..., 1], query_point_2d_int[..., 0]]

                query_point_local = np.concatenate((query_point_2d, query_depths[..., None]), axis=-1)
                raw_slice_data.est_query_point = (
                    np.concatenate((query_frames[..., None], self._local_to_world(query_point_local, raw_slice_data.est_intrinsics[query_frames], raw_slice_data.est_extrinsics[query_frames])), axis=-1)
                ).astype(np.float32)

        # if not np.isfinite(raw_slice_data.est_depths).all():
        #     import ipdb; ipdb.set_trace()
        return RawSliceData(** raw_slice_data.__dict__)

    @abstractmethod
    def _load_slice(
        self, 
        seq_id: int, 
        start: int, 
        length: int, 
        stride: int,
        rng: np.random.Generator,
        executor: ThreadPoolExecutor,
    ) -> RawSliceData:
        pass

    @abstractmethod
    def load_seq_lens(self) -> List[int]:
        pass

    @classmethod
    def from_config(cls, cfg: DictConfig, name: str, override_anno: Optional[str]) -> 'BaseDataProvider':
        if name == 'point_odyssey':
            from datasets.providers.pod_provider import PointOdysseyDataProvider
            return PointOdysseyDataProvider(cfg, override_anno=override_anno)
        elif name == 'dynamic_replica':
            from datasets.providers.dr_provider import DynamicReplicaDataProvider
            return DynamicReplicaDataProvider(cfg, override_anno=override_anno)
        elif name == 'kubric':
            from datasets.providers.kubric_provider import KubricDataProvider
            return KubricDataProvider(cfg, override_anno=override_anno)
        elif name == "tapvid3d":
            from datasets.providers.tapvid3d_provider import TAPVid3dProvider
            return TAPVid3dProvider(cfg, override_anno=override_anno)
        elif name == "custom_kubric":
            from datasets.providers.custom_kubric_provider import CustomKubricDataProvider
            return CustomKubricDataProvider(cfg, override_anno=override_anno)
        elif name == "tapvid":
            from datasets.providers.tapvid_provider import TAPVidProvider
            return TAPVidProvider(cfg, override_anno=override_anno)
        elif name == "dexycb":
            from datasets.providers.dexycb_provider import DexYCBDataProvider
            return DexYCBDataProvider(cfg, override_anno=override_anno)
        elif name == "lsfodyssey":
            from datasets.providers.lsfodyssey_provider import LSFOdysseyProvider
            return LSFOdysseyProvider(cfg, override_anno=override_anno)
        elif name == "iphone":
            from datasets.providers.iphone_provider import IPhoneDataProvider
            return IPhoneDataProvider(cfg, override_anno=override_anno)
        elif name == "bonn":
            from datasets.providers.bonn_provider import BonnDataProvider
            return BonnDataProvider(cfg, override_anno=override_anno)
        else:
            raise ValueError(f"Unknown provider: {name}")

