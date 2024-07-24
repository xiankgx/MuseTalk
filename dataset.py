import glob
import os
import pickle
import time

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def crop_image(image: Image.Image, bbox) -> Image.Image:
    image_np = np.array(image)
    x1, y1, x2, y2 = bbox
    return Image.fromarray(image_np[y1:y2, x1:x2])


def random_mask_for_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    mask_np = np.full((h, w), fill_value=255, dtype=np.uint8)

    y_start = int((0.4 + np.random.rand() * 0.2) * h)
    y_end = int((0.83 + np.random.rand() * (1 - 0.83)) * h)

    x_start = int((0.0 + np.random.rand() * 0.25) * w)
    x_end = int((0.75 + np.random.rand() * 0.25) * w)

    mask_np[y_start:y_end, x_start:x_end] = 0

    return Image.fromarray(mask_np)


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
MASK_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


class HDTFDataset(Dataset):
    def __init__(
        self,
        frame_paths,
        image_transform=IMAGE_TRANSFORM,
        mask_transform=MASK_TRANSFORM,
    ):
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.frame_paths = frame_paths
        # self.data_root = data_root
        # video_dirs = os.listdir(self.data_root)
        # print(f"video_dirs: {video_dirs}")
        # self.video_dirs = video_dirs
        
    def __len__(self):
        # return len(self.video_dirs)
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]

        video_dir = os.path.dirname(os.path.dirname(frame_path))
        # video_dir = os.path.join(self.data_root, self.video_dirs[idx])

        audio_features_count_path = os.path.join(video_dir, "audio_features", "count.pkl")
        with open(audio_features_count_path, "rb") as f:
            num_audio_features = pickle.load(f)

        # tic = time.time()
        coord_list_path = os.path.join(video_dir, "coord_list.pkl")
        with open(coord_list_path, "rb") as f:
            coord_list = pickle.load(f)
            # print(f"coord_list time: {time.time() - tic}")
        num_frames = min(num_audio_features, len(coord_list))

        frame_num = int(os.path.splitext(os.path.basename(frame_path))[0])
        if frame_num >= num_frames:
            frame_num = np.random.randint(num_frames)
        x1, y1, x2, y2 = coord_list[frame_num]
        while (x2 - x1 <= 0) or (y2 - y1 <= 0):
            print("while1")
            frame_num = np.random.randint(num_frames)
            x1, y1, x2, y2 = coord_list[frame_num]

        ref_frame_num = np.random.randint(num_frames)
        x1_ref, y1_ref, x2_ref, y2_ref = coord_list[ref_frame_num]
        while (x2_ref - x1_ref <= 0) or (y2_ref - y1_ref <= 0):
            print("while2")
            ref_frame_num = np.random.randint(num_frames)
            x1_ref, y1_ref, x2_ref, y2_ref = coord_list[ref_frame_num]

        frame_path = os.path.join(video_dir, "frames", f"{frame_num:08d}.jpg")
        ref_frame_path = os.path.join(video_dir, "frames", f"{ref_frame_num:08d}.jpg")
        audio_features_path = os.path.join(video_dir, "audio_features", f"{frame_num:08d}.pkl")
    
        frame = crop_image(Image.open(frame_path), (x1, y1, x2, y2))
        ref_frame = crop_image(
            Image.open(ref_frame_path), (x1_ref, y1_ref, x2_ref, y2_ref)
        )
        mask = random_mask_for_image(frame)

        masked_frame = Image.fromarray(
            (np.array(frame) * (np.array(mask)[..., None] / np.float32(255))).astype(
                np.uint8
            )
        )

        with open(audio_features_path, "rb") as f:
            audio_feature = pickle.load(f)
            # print(f"audio_features time: {time.time() - tic}")

        if self.image_transform:
            ref_frame = self.image_transform(ref_frame)
            frame = self.image_transform(frame)
            masked_frame = self.image_transform(masked_frame)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # ref_image, image, masked_image, masks, audio_feature
        # return {
        #     "ref_image": ref_frame,
        #     "image": frame,
        #     "masked_image": masked_frame,
        #     "mask": mask,
        #     # 10 audio frames, by 1 by 384 * 5 (diff output layers)
        #     "audio_feature": audio_feature.reshape(10, 384 * 5),
        # }
        return ref_frame, frame, masked_frame, mask, audio_feature.reshape(10, 384 * 5)
        # return ref_frame, frame


if __name__ == "__main__":
    frames_paths = glob.glob("HDTF_train_processed/**/frames/*.jpg", recursive=True)
    print(f"# num frames: {len(frames_paths)}")

    ds = HDTFDataset("HDTF_train_processed")
    # sample = ds[np.random.randint(len(ds))]

    
    from tqdm.auto import tqdm
    # dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=8)

    # for batch in tqdm(dl):
    #     continue

    # for sample in tqdm(ds):
    #     print(sample)
    #     continue

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=1)
    for sample in tqdm(dl):
        _ = sample
        # continue

    # print(sample)
    # ref_image = sample["ref_image"]
    # image = sample["image"]
    # masked_image = sample["masked_image"]
    # mask = sample["mask"]
    # audio_feature = sample["audio_feature"]

    # ref_image.save("ref_image.jpg")
    # image.save("image.jpg")
    # masked_image.save("masked_image.jpg")
    # mask.save("mask.png")
    # print(f"audio_feature.shape: {audio_feature.shape}")
