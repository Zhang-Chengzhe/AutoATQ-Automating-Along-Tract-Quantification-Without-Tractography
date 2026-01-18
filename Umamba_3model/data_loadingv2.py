import numpy as np
import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from typing import List, Tuple, Dict


# =========================
# 基础工具
# =========================
def zscore_3d(vol: np.ndarray) -> np.ndarray:
    """对整个 3D volume 做 z-score"""
    return (vol - vol.mean()) / (vol.std() + 1e-8)


def minmax_3d(vol: np.ndarray) -> np.ndarray:
    return (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

def load_img(filepath: str) -> np.ndarray:
    img = nib.load(filepath).get_fdata().astype(np.float32)
    # 将 NaN 替换为 0
    if np.isnan(img).any():
        nan_count = np.isnan(img).sum()
        print(f"[WARN] {filepath}: {nan_count} NaN values replaced with 0")
        img = np.nan_to_num(img, nan=0.0)
    return img


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_files_with_suffix(folder: str, suffix: str) -> List[str]:
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(suffix)])


# =========================
# 2D 切片数据增强（几何+噪声）
# =========================
class SliceAugment2D:
    def __init__(
        self,
        max_rotate_deg: float = 45.0,        # 随机旋转角度范围 [-45°, +45°]
        max_translate_frac: float = 0.10,    # 平移最大幅度（相对 H/W 的 10%）
        resample_scale_range=(0.5, 1.0),     # 重采样降采样系数 s∈[0.5,1]
        p_resample: float = 0.8,             # 执行重采样概率
        noise_mean: float = 0.0,
        noise_std_range=(0.0, 0.05),         # 高斯噪声 std 随机范围
        p_noise: float = 0.5,                # 加噪概率
        fill_img: float = 0.0,               # 几何填充值（图像）
        fill_mask: int = 0                   # 几何填充值（标签）
    ):
        self.max_rotate_deg = max_rotate_deg
        self.max_translate_frac = max_translate_frac
        self.resample_scale_range = resample_scale_range
        self.p_resample = p_resample
        self.noise_mean = noise_mean
        self.noise_std_range = noise_std_range
        self.p_noise = p_noise
        self.fill_img = float(fill_img)
        self.fill_mask = int(fill_mask)

    def _affine_rotate_translate(self, img, mask, angle_deg, tx_px, ty_px):
        # 图像：双线性；标签：最近邻；scale=1, shear=0
        img = TF.affine(
            img, angle=angle_deg, translate=[tx_px, ty_px], scale=1.0, shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR, fill=self.fill_img
        )
        mask = TF.affine(
            mask, angle=angle_deg, translate=[tx_px, ty_px], scale=1.0, shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST, fill=self.fill_mask
        )
        return img, mask

    def _down_up_resample(self, x, scale, is_mask: bool):
        C, H, W = x.shape
        new_h = max(1, int(round(H * scale)))
        new_w = max(1, int(round(W * scale)))
        mode = InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR
        # 降采样
        x_small = TF.resize(x, size=[new_h, new_w], interpolation=mode, antialias=(not is_mask))
        # 还原到原尺寸
        x_back  = TF.resize(x_small, size=[H, W], interpolation=mode, antialias=(not is_mask))
        return x_back

    def _add_gaussian_noise(self, img, mean, std):
        if std <= 0.0:
            return img
        noise = torch.randn_like(img) * std + mean
        return img + noise

    def __call__(self, img_2d: torch.Tensor, mask_2d: torch.Tensor):
        """
        img_2d:  [C_in, H, W]  (float，z-score 后)
        mask_2d: [C_out, H, W] (0/1，多通道)
        """
        assert img_2d.ndim == 3 and mask_2d.ndim == 3, "Expect [C,H,W] tensors."
        _, H, W = img_2d.shape

        # 1) 随机旋转 + 位移（一次仿射，减少插值次数）
        angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
        tx = int(round(random.uniform(-self.max_translate_frac * W, self.max_translate_frac * W)))
        ty = int(round(random.uniform(-self.max_translate_frac * H, self.max_translate_frac * H)))
        img_2d, mask_2d = self._affine_rotate_translate(img_2d, mask_2d, angle_deg=angle, tx_px=tx, ty_px=ty)

        # 2) 重采样（先降再升，模拟低分辨率模糊）
        if random.random() < self.p_resample:
            scale = random.uniform(*self.resample_scale_range)
            img_2d  = self._down_up_resample(img_2d,  scale, is_mask=False)
            mask_2d = self._down_up_resample(mask_2d, scale, is_mask=True)

        # 3) 高斯噪声（仅图像）
        if random.random() < self.p_noise:
            std = random.uniform(*self.noise_std_range)
            img_2d = self._add_gaussian_noise(img_2d, self.noise_mean, std)

        # 最近邻插值理论上保持 0/1，这里做一次稳妥的裁剪/钳位
        mask_2d = mask_2d.clamp(0.0, 1.0)

        return img_2d, mask_2d


# =========================
# 预处理：一次性存成 .npy (+ .npz meta)
# 目录结构：
#   save_dir/
#     images/*.npy
#     labels/*.npy
#     meta/*.npz   (包含 orig_shape, bbox)
# =========================
def preprocess_to_npy(
    data_dir: str,
    label_dir: str,
    save_dir: str,
    crop_bbox: Tuple[int, int, int, int, int, int] = (1, 145, 20, 164, 1, 145),
    dtype=np.float32
):
    """
    将 .nii.gz 批量预处理为 .npy（裁剪+zscore），并保存 meta 以便回原空间。
    """
    ensure_dir(save_dir)
    images_out = os.path.join(save_dir, "images")
    labels_out = os.path.join(save_dir, "labels")
    meta_out   = os.path.join(save_dir, "meta")
    ensure_dir(images_out)
    ensure_dir(labels_out)
    ensure_dir(meta_out)

    img_paths = list_files_with_suffix(data_dir, ".nii.gz")
    lab_paths = list_files_with_suffix(label_dir, ".nii.gz")
    assert len(img_paths) == len(lab_paths), "images 与 labels 数量不一致！"

    d0, d1, h0, h1, w0, w1 = crop_bbox

    for img_path, lab_path in zip(img_paths, lab_paths):
        base = os.path.basename(img_path).replace(".nii.gz", "")
        print(f"[Preprocess] {base}")

        # 这里直接用 nib.load 拿 affine
        img_nii = nib.load(img_path)
        lab_nii = nib.load(lab_path)
        img_full = img_nii.get_fdata().astype(np.float32)
        lab_full = lab_nii.get_fdata().astype(np.float32)
        affine = img_nii.affine  # 假设图像与标签同一空间

        # 统一补通道维
        if img_full.ndim == 3: img_full = img_full[..., np.newaxis]
        if lab_full.ndim == 3: lab_full = lab_full[..., np.newaxis]

        D0,H0,W0,_ = img_full.shape

        # 裁剪
        img = img_full[d0:d1, h0:h1, w0:w1, :]
        lab = lab_full[d0:d1, h0:h1, w0:w1, :]

        # 每个通道归一化处理
        for c in range(img.shape[-1]):
            img[..., c] = zscore_3d(img[..., c])

        # 保存 .npy
        np.save(os.path.join(images_out, f"{base}.npy"), img.astype(dtype))
        np.save(os.path.join(labels_out, f"{base}.npy"), lab.astype(dtype))

        # 保存 meta
        np.savez_compressed(
            os.path.join(meta_out, f"{base}.npz"),
            orig_shape=np.array([D0, H0, W0], dtype=np.int32),
            bbox=np.array([d0, d1, h0, h1, w0, w1], dtype=np.int32),
            affine=affine.astype(np.float64)
        )


# =========================
# 工具：根据方向抽 2D 切片
# =========================
def random_2Dimage(image: np.ndarray, label: np.ndarray, slice_idx: int, direction: str):
    """
    image: (D,H,W,Cin)
    label: (D,H,W,Cout)
    返回某一方向的 2D 切片：(H,W,Cin)、(H,W,Cout)
    """
    if direction == 'sagittal':
        image_slices = image[slice_idx, :, :, :]  # (H,W,Cin)
        label_slices = label[slice_idx, :, :, :]  # (H,W,Cout)
    elif direction == 'coronal':
        image_slices = image[:, slice_idx, :, :]
        label_slices = label[:, slice_idx, :, :]
    elif direction == 'axial':
        image_slices = image[:, :, slice_idx, :]
        label_slices = label[:, :, slice_idx, :]
    return image_slices, label_slices


# =========================
# Dataset：支持 .nii.gz 与 .npy
# - 若 data_dir/labels_dir 下发现 .npy，则直接加载 .npy（最快）
# - 否则走 .nii.gz -> 裁剪+zscore 路线
# =========================
class MyDataset(Dataset):
    def __init__(self, data_dir, labels_dir, transform=None, direction=None, augment: bool=False, use_mmap_npy: bool=False):
        """
        data_dir, labels_dir:
          - 方案A（推荐）：预处理后的目录结构
              data_dir   = ".../dataset_npy/fold1/images"
              labels_dir = ".../dataset_npy/fold1/labels"
              （可选）同级还有 meta/ 存 orig_shape 与 bbox
          - 方案B：原始的 .nii.gz 目录
              data_dir   = ".../dataset/images/fold1"
              labels_dir = ".../dataset/labels/fold1"
        """
        self.transform = transform
        self.augment = augment
        self.use_mmap_npy = use_mmap_npy  # 仅对 .npy 有效，减少内存峰值
        self.direction = direction

        # 识别数据源类型
        self.npy_mode = False
        npy_imgs = list_files_with_suffix(data_dir, ".npy")
        npy_labs = list_files_with_suffix(labels_dir, ".npy")
        nii_imgs = list_files_with_suffix(data_dir, ".nii.gz")
        nii_labs = list_files_with_suffix(labels_dir, ".nii.gz")

        if len(npy_imgs) > 0 and len(npy_labs) > 0:
            # .npy 模式（最快）
            self.npy_mode = True
            self.data_path = npy_imgs
            self.labels_path = npy_labs
            # 尝试定位 meta 目录（与 images/labels 同级的 meta/）
            images_parent = os.path.dirname(data_dir.rstrip("/"))
            candidate_meta = os.path.join(images_parent, "meta")
            self.meta_dir = candidate_meta if os.path.isdir(candidate_meta) else None

        else:
            # .nii.gz 模式（兼容旧流程）
            assert len(nii_imgs) > 0 and len(nii_labs) > 0, \
                f"在 {data_dir} / {labels_dir} 未找到 .npy 或 .nii.gz 数据"
            self.npy_mode = False
            self.data_path = nii_imgs
            self.labels_path = nii_labs
            self.meta_dir = None

        # 统一配准窗口（与原代码一致）
        self.crop_bbox = (1, 145, 20, 164, 1, 145)
        d0, d1, h0, h1, w0, w1 = self.crop_bbox

        # 预加载 or 仅保存路径
        # 为了与原训练代码兼容，这里仍默认预加载到内存（.npy 时速度很快）
        # 如若内存吃紧，可按需改为懒加载。
        self.all_imgs: List[np.ndarray] = []
        self.all_labels: List[np.ndarray] = []
        self.meta: List[Dict] = []

        for img_path, lab_path in zip(self.data_path, self.labels_path):
            base = os.path.basename(img_path).replace(".npy", "").replace(".nii.gz", "")

            # 先给 affine 一个缺省值，避免未定义
            affine = None

            if self.npy_mode:
                # .npy：已裁剪数据
                if self.use_mmap_npy:
                    img = np.load(img_path, mmap_mode='r'); lab = np.load(lab_path, mmap_mode='r')
                    img = np.asarray(img, dtype=np.float32).copy()
                    lab = np.asarray(lab, dtype=np.float32).copy()
                else:
                    img = np.load(img_path).astype(np.float32)
                    lab = np.load(lab_path).astype(np.float32)

                if self.meta_dir:
                    meta_file = os.path.join(self.meta_dir, f"{base}.npz")
                    if os.path.isfile(meta_file):
                        m = np.load(meta_file)
                        orig_shape = tuple(map(int, m["orig_shape"]))
                        bbox = tuple(map(int, m["bbox"]))
                        # 只有当 npz 里确实存了 affine 时才读取，否则保持 None
                        affine = m["affine"] if "affine" in m.files else None
                    else:
                        orig_shape = tuple(img.shape[:3])
                        bbox = self.crop_bbox
                        affine = None
                else:
                    orig_shape = tuple(img.shape[:3])
                    bbox = self.crop_bbox
                    affine = None

            else:
                # .nii.gz：读取 -> 裁剪 -> 归一化
                img_full = load_img(img_path)
                lab_full = load_img(lab_path)

                # 这里直接从原始 NIfTI 取 affine（假设图像与标签同一空间）
                try:
                    affine = nib.load(img_path).affine
                except Exception:
                    affine = None

                if img_full.ndim == 4:
                    D0, H0, W0, _ = img_full.shape
                else:
                    D0, H0, W0 = img_full.shape

                d0, d1, h0, h1, w0, w1 = self.crop_bbox
                # 若是 3D，补通道维，防止切片时索引报错
                if img_full.ndim == 3: img_full = img_full[..., np.newaxis]
                if lab_full.ndim == 3: lab_full = lab_full[..., np.newaxis]

                img = img_full[d0:d1, h0:h1, w0:w1, :]
                lab = lab_full[d0:d1, h0:h1, w0:w1, :]
                for c in range(img.shape[-1]):
                    img[..., c] = zscore_3d(img[..., c])

                orig_shape = (int(D0), int(H0), int(W0))
                bbox = self.crop_bbox

            self.all_imgs.append(img.astype(np.float32))
            self.all_labels.append(lab.astype(np.float32))
            self.meta.append({"orig_shape": orig_shape, "bbox": bbox, "affine": affine})

        # 增强器（仅在训练-切片模式下使用）
        self.slice_aug = SliceAugment2D(
            max_rotate_deg=45.0,
            max_translate_frac=0.10,
            resample_scale_range=(0.5, 0.9),
            p_resample=0.8,
            noise_mean=0.0,
            noise_std_range=(0.0, 0.05),
            p_noise=0.8,
            fill_img=0.0,
            fill_mask=0
        )

    def __len__(self):
        # 训练/切片模式：每个方向取 144 张切片
        return len(self.data_path) * 144


    def __getitem__(self, idx):
        subj_idx = idx // 144
        slice_idx = idx % 144
        vol = self.all_imgs[subj_idx]       # (D,H,W,Cin) -> (144,144,144,Cin)
        lab_vol = self.all_labels[subj_idx] # (D,H,W,Cout)

        if self.direction == 'sagittal':
            image_slices, label_slices = random_2Dimage(vol, lab_vol, slice_idx, 'sagittal')
        elif self.direction == 'coronal':
            image_slices, label_slices = random_2Dimage(vol, lab_vol, slice_idx, 'coronal')
        elif self.direction == 'axial':
            image_slices, label_slices = random_2Dimage(vol, lab_vol, slice_idx, 'axial')

        # [C,H,W]
        image = np.transpose(image_slices, (2, 0, 1))
        labels = np.transpose(label_slices, (2, 0, 1))

        image = torch.from_numpy(image).float()
        labels = torch.from_numpy(labels).float()

        # 数据增强（仅训练）
        if self.augment:
            image, labels = self.slice_aug(image, labels)

        if self.transform is not None:
            image = self.transform(image)


        return image, labels


# =========================
# 直接运行：示例
# =========================
if __name__ == '__main__':
    # 示例：把 fold1 的 .nii.gz 预处理为 .npy
    # preprocess_to_npy(
    #     data_dir='/data/zcz/test_1010/dataset/images/test',
    #     label_dir='/data/zcz/test_1010/dataset/labels/test',
    #     save_dir='/data/zcz/test_1010/dataset_npy/fold1'
    # )

    # 示例：加载 .npy 进行训练切片
    data_dir_npy = '/data/zcz/test_1010/dataset/images/test'
    label_dir_npy = '/data/zcz/test_1010/dataset/labels/test'
    dataset = MyDataset(data_dir_npy, label_dir_npy, direction='axial', augment=True, use_mmap_npy=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    for img, mask in dataloader:
        print("Image:", img.shape)  # torch.Size([B, Cin, H, W])
        print("Mask:", mask.shape)  # torch.Size([B, Cout, H, W])
        break
