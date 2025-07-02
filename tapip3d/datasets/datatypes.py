from typing import Any, List, Literal, Optional, Union
import torch
import numpy as np
import dataclasses
from einops import rearrange
from ..datasets.utils.geometry import batch_distance_to_depth
import copy

EPS = 1e-6

@dataclasses.dataclass
class RawSliceData:
    seq_name: str
    seq_id: int
    visibs: np.ndarray # (t, n)
    valids: np.ndarray # (t, n)
    rgbs: np.ndarray # (t, h, w, 3)

    orig_resolution: np.ndarray # (2)

    gt_intrinsics: np.ndarray # (t, 3, 3)
    gt_extrinsics: np.ndarray # (t, 4, 4)
    gt_trajs_3d: np.ndarray # (t, n, 3)
    gt_depths: np.ndarray # (t, h, w)

    est_intrinsics: np.ndarray # (t, 3, 3)
    est_extrinsics: np.ndarray # (t, 4, 4)
    est_trajs_3d: np.ndarray # (t, n, 3)
    est_depths: np.ndarray # (t, h, w)
    same_scale: bool

    gt_depth_roi: Optional[np.ndarray] = None # (2,)
    est_depth_roi: Optional[np.ndarray] = None # (2,)

    segmentation: Optional[np.ndarray] = None # (t, h, w)
    
    gt_query_point: Optional[np.ndarray] = None # (n, 3)
    est_query_point: Optional[np.ndarray] = None # (n, 3)

    flags: List[str] = dataclasses.field(default_factory=list)
    skip_post_init: bool = False
    annotated: bool = False

    @classmethod
    def create(cls, copy_gt_to_est: bool = False, **kwargs) -> 'RawSliceData':
        kwargs = kwargs.copy()
        if copy_gt_to_est:
            if not 'same_scale' in kwargs:
                kwargs['same_scale'] = True
            for key in kwargs.copy():
                if key.startswith('gt_'):
                    kwargs[key.replace('gt_', 'est_')] = copy.deepcopy(kwargs[key])
        return cls(**kwargs)

    def __post_init__(self):
        if self.skip_post_init:
            return
        assert isinstance(self.seq_name, str), "seq_name must be a string"
        assert isinstance(self.seq_id, int), "seq_id must be an integer"

        assert isinstance(self.visibs, np.ndarray) and self.visibs.dtype == np.bool_, "visibs must be a boolean numpy array"
        assert isinstance(self.valids, np.ndarray) and self.valids.dtype == np.bool_, "valids must be a boolean numpy array"
        assert isinstance(self.rgbs, np.ndarray) and self.rgbs.dtype == np.uint8, "rgbs must be a uint8 numpy array"
        assert isinstance(self.orig_resolution, np.ndarray) and self.orig_resolution.dtype == np.int32, "orig_resolution must be a int32 numpy array"
        if self.segmentation is not None:
            assert isinstance(self.segmentation, np.ndarray) and self.segmentation.dtype == np.int32, "segmentation must be a uint8 numpy array"

        assert len(self.rgbs.shape) == 4 and self.rgbs.shape[-1] == 3, "the shape of rgbs must be (t, h, w, 3)"
        assert len(self.orig_resolution.shape) == 1 and self.orig_resolution.shape[0] == 2, "the shape of orig_resolution must be (2,)"

        # gt
        assert isinstance(self.gt_depths, np.ndarray) and self.gt_depths.dtype == np.float32, "gt_depths must be a float32 numpy array"
        assert isinstance(self.gt_intrinsics, np.ndarray) and self.gt_intrinsics.dtype == np.float32, "gt_intrinsics must be a float32 numpy array"
        assert isinstance(self.gt_extrinsics, np.ndarray) and self.gt_extrinsics.dtype == np.float32, "gt_extrinsics must be a float32 numpy array"
        assert isinstance(self.gt_trajs_3d, np.ndarray) and self.gt_trajs_3d.dtype == np.float32, "gt_trajs_3d must be a float32 numpy array"
        assert len(self.gt_intrinsics.shape) == 3 and self.gt_intrinsics.shape[-2:] == (3, 3), "the shape of gt_intrinsics must be (t, 3, 3)"
        assert len(self.gt_depths.shape) == 3, "the shape of gt_depths must be (t, h, w)"
        if self.gt_query_point is not None:
            assert isinstance(self.gt_query_point, np.ndarray) and self.gt_query_point.dtype == np.float32, "gt_query_point must be a float32 numpy array"

        # est
        assert isinstance(self.est_depths, np.ndarray) and self.est_depths.dtype == np.float32, "est_depths must be a float32 numpy array"
        assert isinstance(self.est_intrinsics, np.ndarray) and self.est_intrinsics.dtype == np.float32, "est_intrinsics must be a float32 numpy array"
        assert isinstance(self.est_extrinsics, np.ndarray) and self.est_extrinsics.dtype == np.float32, "est_extrinsics must be a float32 numpy array"
        assert isinstance(self.est_trajs_3d, np.ndarray) and self.est_trajs_3d.dtype == np.float32, "est_trajs_3d must be a float32 numpy array"
        assert len(self.est_intrinsics.shape) == 3 and self.est_intrinsics.shape[-2:] == (3, 3), "the shape of est_intrinsics must be (t, 3, 3)"
        assert len(self.est_depths.shape) == 3, "the shape of est_depths must be (t, h, w)"
        if self.est_query_point is not None:
            assert isinstance(self.est_query_point, np.ndarray) and self.est_query_point.dtype == np.float32, "est_query_point must be a float32 numpy array"

        assert id(self.gt_extrinsics) != id(self.est_extrinsics), "gt_extrinsics and est_extrinsics must be different objects"
        assert id(self.gt_intrinsics) != id(self.est_intrinsics), "gt_intrinsics and est_intrinsics must be different objects"
        assert id(self.gt_trajs_3d) != id(self.est_trajs_3d), "gt_trajs_3d and est_trajs_3d must be different objects"
        assert id(self.gt_depths) != id(self.est_depths), "gt_depths and est_depths must be different objects"
        if self.gt_query_point is not None and self.est_query_point is not None:
            assert id(self.gt_query_point) != id(self.est_query_point), "gt_query_point and est_query_point must be different objects"

        assert self.rgbs.shape[:3] == self.est_depths.shape[:3], "rgbs and est_depths must have the same size"
        assert self.est_depths.shape == self.gt_depths.shape, "est_depths and gt_depths must have the same shape"

        assert not (self.visibs & ~self.valids).any(), "visible points should be valid"

        if self.gt_depth_roi is not None:
            assert isinstance(self.gt_depth_roi, np.ndarray) and self.gt_depth_roi.dtype == np.float32, "gt_depth_roi must be a float32 numpy array"
            assert self.gt_depth_roi.shape == (2,), "the shape of gt_depth_roi must be (2,)"
        if self.est_depth_roi is not None:
            assert isinstance(self.est_depth_roi, np.ndarray) and self.est_depth_roi.dtype == np.float32, "est_depth_roi must be a float32 numpy array"
            assert self.est_depth_roi.shape == (2,), "the shape of est_depth_roi must be (2,)"

    def copy(self) -> 'RawSliceData':
        return RawSliceData(**self.__dict__)

@dataclasses.dataclass
class SliceData:
    seq_name: Union[str, List[str]]
    seq_id: torch.Tensor # (b)
    sample_id: torch.Tensor # (b)
    visibs: torch.Tensor # (t, n)
    valids: torch.Tensor # (t, n)
    rgbs: torch.Tensor # (t, 3, h, w)
    orig_resolution: torch.Tensor # (2)

    gt_query_point: torch.Tensor # (n, 3)
    gt_intrinsics: torch.Tensor # (t, 3, 3)
    gt_extrinsics: torch.Tensor # (t, 4, 4)
    gt_trajs_3d: torch.Tensor # (t, n, 3)
    gt_depths: torch.Tensor # (t, h, w)

    est_query_point: torch.Tensor # (n, 3)
    est_intrinsics: torch.Tensor # (t, 3, 3)
    est_extrinsics: torch.Tensor # (t, 4, 4)
    est_trajs_3d: torch.Tensor # (t, n, 3)
    est_depths: torch.Tensor # (t, h, w)

    same_scale: bool
    est_depth_roi: Optional[torch.Tensor] = None # (2,)
    gt_depth_roi: Optional[torch.Tensor] = None # (2,)
    annot_mode: Literal['gt', 'est', None] = None
    segmentation: Optional[torch.Tensor] = None # (t, h, w)

    flags: List[str] = dataclasses.field(default_factory=list)
    skip_post_init: bool = False
    
    @torch.no_grad()
    def __post_init__(self):
        if self.skip_post_init:
            return
        assert isinstance(self.seq_name, (str, list)), "seq_name must be a string or a list of strings"
        assert isinstance(self.seq_id, torch.Tensor) and self.seq_id.dtype == torch.int32, "seq_id must be a int32 tensor"
        assert isinstance(self.sample_id, torch.Tensor) and self.sample_id.dtype == torch.int32, "sample_id must be a int32 tensor"
        assert isinstance(self.visibs, torch.Tensor) and self.visibs.dtype == torch.bool, "visibs must be a boolean tensor"
        assert isinstance(self.valids, torch.Tensor) and self.valids.dtype == torch.bool, "valids must be a boolean tensor"
        assert isinstance(self.rgbs, torch.Tensor) and self.rgbs.dtype == torch.float32, "rgbs must be a float32 tensor"
        assert isinstance(self.orig_resolution, torch.Tensor) and self.orig_resolution.dtype == torch.int32, "orig_resolution must be a int32 tensor"

        if len(self.rgbs.shape) == 4:
            assert self.rgbs.shape[1] == 3 and len(self.rgbs.shape) == 4, "the shape of rgbs must be (t, 3, h, w)"
        else:
            assert self.rgbs.shape[2] == 3 and len(self.rgbs.shape) == 5, "the shape of rgbs must be (b, t, 3, h, w)"
        assert self.rgbs.max() <= 1.0 + EPS and self.rgbs.min() >= -EPS, "rgbs must be in the range of [0, 1]"

        assert isinstance(self.gt_trajs_3d, torch.Tensor) and self.gt_trajs_3d.dtype == torch.float32, "gt_trajs_3d must be a float32 tensor"
        assert isinstance(self.est_trajs_3d, torch.Tensor) and self.est_trajs_3d.dtype == torch.float32, "est_trajs_3d must be a float32 tensor"
        
        assert isinstance(self.gt_depths, torch.Tensor) and self.gt_depths.dtype == torch.float32, "gt_depths must be a float32 tensor"
        assert isinstance(self.est_depths, torch.Tensor) and self.est_depths.dtype == torch.float32, "est_depths must be a float32 tensor"
        
        assert isinstance(self.gt_intrinsics, torch.Tensor) and self.gt_intrinsics.dtype == torch.float32, "gt_intrinsics must be a float32 tensor"
        assert isinstance(self.gt_extrinsics, torch.Tensor) and self.gt_extrinsics.dtype == torch.float32, "gt_extrinsics must be a float32 tensor"
        assert isinstance(self.est_intrinsics, torch.Tensor) and self.est_intrinsics.dtype == torch.float32, "est_intrinsics must be a float32 tensor"
        assert isinstance(self.est_extrinsics, torch.Tensor) and self.est_extrinsics.dtype == torch.float32, "est_extrinsics must be a float32 tensor"
        assert isinstance(self.gt_query_point, torch.Tensor) and self.gt_query_point.dtype == torch.float32, "gt_query_point must be a float32 tensor"
        assert isinstance(self.est_query_point, torch.Tensor) and self.est_query_point.dtype == torch.float32, "est_query_point must be a float32 tensor"

        if self.segmentation is not None:
            assert isinstance(self.segmentation, torch.Tensor) and self.segmentation.dtype == torch.int32, "segmentation must be a int32 tensor"

        assert torch.allclose(self.gt_query_point[..., 0], torch.round(self.gt_query_point[..., 0])), "the first dimension of gt_query_point must be integer"

        assert torch.allclose(self.est_query_point[..., 0], torch.round(self.est_query_point[..., 0])), "the first dimension of est_query_point must be integer"
        assert torch.allclose(self.gt_query_point[..., 0], self.est_query_point[..., 0]), "gt_query_point and est_query_point must have the same query frame"
        
        # import ipdb; ipdb.set_trace()
        if not torch.isfinite(self.est_depths).all():
            # import ipdb; ipdb.set_trace()
            # self.est_depths.zero_()
            raise Exception("est_depths is not finite")
            import ipdb; ipdb.set_trace()
        assert torch.isfinite(self.gt_depths).all(), "gt_depths must be finite"
        assert torch.isfinite(self.est_depths).all(), "est_depths must be finite"
        assert self.gt_depths.min() >= 0.0, "gt_depths must be non-negative"
        assert self.est_depths.min() >= 0.0, "est_depths must be non-negative"

        assert self.rgbs.shape[-2:] == self.est_depths.shape[-2:], "rgbs and est_depths must have the same size"
        assert self.est_depths.shape == self.gt_depths.shape, "est_depths and gt_depths must have the same shape"

        if self.gt_depth_roi is not None:
            assert self.gt_depth_roi.shape[-1] == 2, "the shape of gt_depth_roi must be (2,) or (b, 2)"
        if self.est_depth_roi is not None:
            assert self.est_depth_roi.shape[-1] == 2, "the shape of est_depth_roi must be (2,) or (b, 2)"

    @staticmethod
    def collate(batch: List['SliceData']) -> 'SliceData':
        kwargs = {}
        for field in dataclasses.fields(SliceData):
            if isinstance(getattr(batch[0], field.name), torch.Tensor):
                kwargs[field.name] = torch.stack([getattr(item, field.name) for item in batch])
            elif field.name == 'segmentation' and batch[0].segmentation is None:
                assert all(item.segmentation is None for item in batch)
                kwargs[field.name] = None
            elif field.name in ['annot_mode', 'same_scale', 'flags', 'skip_post_init']:
                assert all(getattr(item, field.name) == batch[0].__dict__[field.name] for item in batch), "all items in the batch must have the same value for the field"
                kwargs[field.name] = getattr(batch[0], field.name)
            elif field.name == 'gt_depth_roi' or field.name == 'est_depth_roi':
                if all(getattr(item, field.name) is None for item in batch):
                    kwargs[field.name] = None
                else:
                    assert all(getattr(item, field.name) is not None for item in batch), "all items in the batch must either be None or not None"
                    kwargs[field.name] = torch.stack([getattr(item, field.name) for item in batch])
            else:
                kwargs[field.name] = [getattr(item, field.name) for item in batch]
        return SliceData(**kwargs)

    def cuda(self) -> 'SliceData':
        def _to_cuda(x):
            if isinstance(x, torch.Tensor):
                return x.cuda()
            return x
        return SliceData(**{k: _to_cuda(v) for k, v in self.__dict__.items()}) # type: ignore
    
    def to(self, device: Union[str, torch.device]) -> 'SliceData':
        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x
        return SliceData(**{k: _to_device(v) for k, v in self.__dict__.items()}) # type: ignore

    def copy(self) -> 'SliceData':
        d = self.__dict__.copy()
        if "skip_post_init" in d:
            d["skip_post_init"] = True
        r = SliceData(**d)
        r.skip_post_init = self.skip_post_init
        return r

    def time_slice(self, start: int, end: int) -> 'SliceData':
        if len(self.rgbs.shape) == 5:
            assert start >= 0 and end <= self.rgbs.shape[1] and start < end, "the range of start and end is out of bounds"
            return SliceData(
                seq_name=self.seq_name,
                seq_id=self.seq_id,
                sample_id=self.sample_id,
                gt_trajs_3d=self.gt_trajs_3d[:, start:end],
                est_trajs_3d=self.est_trajs_3d[:, start:end],
                visibs=self.visibs[:, start:end],
                valids=self.valids[:, start:end],
                rgbs=self.rgbs[:, start:end],
                gt_depths=self.gt_depths[:, start:end],
                est_depths=self.est_depths[:, start:end],
                gt_query_point=self.gt_query_point,
                est_query_point=self.est_query_point,
                gt_intrinsics=self.gt_intrinsics[:, start:end],
                est_intrinsics=self.est_intrinsics[:, start:end],
                gt_extrinsics=self.gt_extrinsics[:, start:end],
                est_extrinsics=self.est_extrinsics[:, start:end],
                orig_resolution=self.orig_resolution,
                annot_mode=self.annot_mode,
                same_scale=self.same_scale,
                skip_post_init=self.skip_post_init,
                flags=self.flags,
                gt_depth_roi=self.gt_depth_roi,
                est_depth_roi=self.est_depth_roi,
            )
        elif len(self.rgbs.shape) == 4:
            assert start >= 0 and end <= self.rgbs.shape[0] and start < end, "the range of start and end is out of bounds"
            return SliceData(
                seq_name=self.seq_name,
                seq_id=self.seq_id,
                sample_id=self.sample_id,
                gt_trajs_3d=self.gt_trajs_3d[start:end],
                est_trajs_3d=self.est_trajs_3d[start:end],
                visibs=self.visibs[start:end],
                valids=self.valids[start:end],
                rgbs=self.rgbs[start:end],
                gt_depths=self.gt_depths[start:end],
                est_depths=self.est_depths[start:end],
                gt_query_point=self.gt_query_point,
                est_query_point=self.est_query_point,
                gt_intrinsics=self.gt_intrinsics[start:end],
                est_intrinsics=self.est_intrinsics[start:end],
                gt_extrinsics=self.gt_extrinsics[start:end],
                est_extrinsics=self.est_extrinsics[start:end],
                orig_resolution=self.orig_resolution,
                annot_mode=self.annot_mode,
                same_scale=self.same_scale,
                skip_post_init=self.skip_post_init,
                flags=self.flags,
                gt_depth_roi=self.gt_depth_roi,
                est_depth_roi=self.est_depth_roi,
            )
        else:
            raise ValueError(f"Unknown shape of rgbs: {self.rgbs.shape}")

    def with_annot_mode(self, annot_mode: Literal['gt', 'est', None]) -> 'SliceData':
        assert annot_mode in ['gt', 'est', None], "annot_mode must be either 'gt', 'est', or None"
        ret = self.copy()
        ret.annot_mode = annot_mode
        return ret

    def __getattr__(self, name: str) -> Any:
        if not f"gt_{name}" in self.__dict__:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if self.annot_mode is None:
            raise ValueError(f"You must call with_annot_mode(annot_mode) before accessing {name}")
        if self.annot_mode == 'gt':
            return getattr(self, f"gt_{name}")
        else:
            return getattr(self, f"est_{name}")

    def with_causal_mask(self) -> 'SliceData':
        sample = self.copy()

        query_frame = sample.gt_query_point[..., 0].long()
        eye = torch.eye(sample.valids.shape[-2], dtype=query_frame.dtype, device=query_frame.device)
        query_frame_to_eval_frames = torch.cumsum(eye, dim=1)
        mask = rearrange(query_frame_to_eval_frames[query_frame] > 0, 'b n t -> b t n')
        
        sample.valids = sample.valids & mask

        return sample