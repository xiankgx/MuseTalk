import glob
import os
import pickle
import time

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from musetalk.utils.preprocessing import create_mask_from_2dlmk, extract_mouth_landmarks, extract_lower_face_landmarks

IMAGE_SIZE = (256, 256)
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)


def crop_image(image: Image.Image, bbox) -> Image.Image:
    image_np = np.array(image)
    x1, y1, x2, y2 = bbox
    return Image.fromarray(image_np[y1:y2, x1:x2])


def random_mask_for_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    mask_np = np.full((h, w), fill_value=255, dtype=np.uint8)

    y_start = int((0.4 + np.random.rand() * 0.1) * h)
    y_end = int((0.83 + np.random.rand() * (1 - 0.83)) * h)

    x_start = int((0.0 + np.random.rand() * 0.2) * w)
    x_end = int((0.8 + np.random.rand() * 0.2) * w)

    mask_np[y_start:y_end, x_start:x_end] = 0

    return Image.fromarray(mask_np)


def half_mask_for_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    mask_np = np.full((h, w), fill_value=255, dtype=np.uint8)

    mask_np[h//2:, :] = 0

    return Image.fromarray(mask_np)


def get_mouth_bounding_box(mouth_landmark_list):
    mouth_landmarks = np.array(mouth_landmark_list)
    # print(f"mouth_landmarks.shape: {mouth_landmarks.shape}")
    x1, y1 = mouth_landmarks.min(axis=0)
    x2, y2 = mouth_landmarks.max(axis=0)
    return x1, y1, x2, y2


def mouth_mask_for_image(image: Image.Image, mouth_bbox, crop_bbox):
    w, h = image.size
    x1, y1, x2, y2 = mouth_bbox
    mouth_bbox_h = y2 - y1
    mouth_bbox_w = x2 - x1
    mask_np = np.full((h, w), fill_value=255, dtype=np.uint8)
    mask_np[
        max(y1 - int(mouth_bbox_h * (0.3 + np.random.rand() * 0.3)), 0) : min(
            y2 + int(mouth_bbox_h * (0.3 + np.random.rand() * 0.3)), h
        ),
        max(x1 - int(mouth_bbox_w * (0.3 + np.random.rand() * 0.3)), 0) : min(
            x2 + int(mouth_bbox_w * (0.3 + np.random.rand() * 0.3)), w
        ),
    ] = 0
    x1, y1, x2, y2 = crop_bbox
    return Image.fromarray(mask_np[y1:y2, x1:x2])


def random_mask_erotion_dilation(mask: Image.Image, kernel_size=(5, 5), iterations: int=7) -> Image.Image:
    if iterations == 0:
        return mask

    mask_np = np.array(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # if np.random.rand() < 0.33:
    #     dilated_mask = cv2.erode(mask_np, kernel, iterations=iterations)
    # else:
    dilated_mask = cv2.dilate(mask_np, kernel, iterations=iterations)
    return Image.fromarray(dilated_mask)


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ]
)
MASK_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
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

        audio_features_count_path = os.path.join(
            video_dir, "audio_features", "count.pkl"
        )
        with open(audio_features_count_path, "rb") as f:
            num_audio_features = pickle.load(f)

        # tic = time.time()
        coord_list_path = os.path.join(video_dir, "coord_list.pkl")
        with open(coord_list_path, "rb") as f:
            coord_list = pickle.load(f)
            # print(f"coord_list time: {time.time() - tic}")

        # mouth_landmarks_list_path = os.path.join(video_dir, "mouth_landmarks_list.pkl")
        # with open(mouth_landmarks_list_path, "rb") as f:
        #     mouth_landmarks_list = pickle.load(f)
        # assert len(mouth_landmarks_list) == len(coord_list)
        # # print(f"mouth_landmarks_list: {mouth_landmarks_list}")

        facial_landmarks_path = os.path.join(video_dir, "facial_landmarks.pkl")
        with open(facial_landmarks_path, "rb") as f:
            facial_landmarks = pickle.load(f)
            if isinstance(facial_landmarks, list):
                facial_landmarks = np.stack(facial_landmarks, axis=0)
        assert len(facial_landmarks) == len(coord_list)
        # print(f"mouth_landmarks_list: {mouth_landmarks_list}")

        num_frames = min(num_audio_features, len(coord_list))

        frame_num = int(os.path.splitext(os.path.basename(frame_path))[0])
        if frame_num >= num_frames:
            print(
                f"Picking another frame as current frame because there is no corresponding audio feature for this frame. current frame: {frame_num}, num_audio_features: {num_audio_features}: len(coord_list): {len(coord_list)}, num_frames: {num_frames}"
            )
            frame_num = np.random.randint(num_frames)
        x1, y1, x2, y2 = coord_list[frame_num]
        while (x2 - x1 <= 0) or (y2 - y1 <= 0):
            print(
                "Picking another frame as current frame due to invalid face bounding box."
            )
            frame_num = np.random.randint(num_frames)
            x1, y1, x2, y2 = coord_list[frame_num]

        ref_frame_num = np.random.randint(num_frames)
        x1_ref, y1_ref, x2_ref, y2_ref = coord_list[ref_frame_num]
        while (x2_ref - x1_ref <= 0) or (y2_ref - y1_ref <= 0):
            print(
                "Picking another frame as ref frame due to invalid face bounding box."
            )
            ref_frame_num = np.random.randint(num_frames)
            x1_ref, y1_ref, x2_ref, y2_ref = coord_list[ref_frame_num]

        frame_path = os.path.join(video_dir, "frames", f"{frame_num:08d}.jpg")
        ref_frame_path = os.path.join(video_dir, "frames", f"{ref_frame_num:08d}.jpg")
        audio_features_path = os.path.join(
            video_dir, "audio_features", f"{frame_num:08d}.pkl"
        )

        frame = crop_image(Image.open(frame_path), (x1, y1, x2, y2))
        ref_frame = crop_image(
            Image.open(ref_frame_path), (x1_ref, y1_ref, x2_ref, y2_ref)
        )

        # mask = half_mask_for_image(frame)
        # mask = random_mask_for_image(frame)
        # mask = mouth_mask_for_image(
        #     Image.open(frame_path),
        #     crop_bbox=(x1, y1, x2, y2),
        #     mouth_bbox=get_mouth_bounding_box(mouth_landmarks_list[frame_num]),
        # )


        try:
            # print(f"facial_landmarks.shape: {facial_landmarks.shape}")
            lower_face_landmarks = extract_lower_face_landmarks(facial_landmarks)[frame_num]
            # print(f"lower_face_landmarks.shape: {lower_face_landmarks.shape}")

            mask = np.array(crop_image(create_mask_from_2dlmk(np.array(Image.open(frame_path)), lower_face_landmarks), (x1, y1, x2, y2)))
            if mask.ndim == 3:
                mask = mask[..., 0]
            # Invert the mask
            mask = Image.fromarray(255 - mask)
            mask = random_mask_erotion_dilation(mask, iterations=np.random.randint(0, 9+1))

            if np.random.rand() <= 0.7:
                if np.random.rand() <= 0.7:
                    mask = half_mask_for_image(frame)
                else:
                    mask = random_mask_for_image(frame)
        except Exception as e:
            print(f"Error obtaining lower face landmark: {str(e)}")
            if np.random.rand() <= 0.7:
                mask = half_mask_for_image(frame)
            else:
                mask = random_mask_for_image(frame)
        
        # mask = crop_image(create_mask_from_2dlmk(Image.open(frame_path), extract_lower_face_landmarks(facial_landmarks)), (x1, y1, x2, y2))

        # if np.random.rand() <= 0.9:
        #     if np.random.rand() <= 0.7:
        #         mask = half_mask_for_image(frame)
        #     else:
        #         mask = random_mask_for_image(frame)
        # else:
        #     mask = mouth_mask_for_image(
        #         Image.open(frame_path),
        #         crop_bbox=(x1, y1, x2, y2),
        #         mouth_bbox=get_mouth_bounding_box(mouth_landmarks_list[frame_num]),
        #     )

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
        return ref_frame, frame, masked_frame, mask, audio_feature.reshape(-1, 5 * 384)[[4, 5]]  # XXX
        # return ref_frame, frame, masked_frame, mask, audio_feature


if __name__ == "__main__":
    from tqdm.auto import tqdm
    from torch.utils.data import DataLoader

    frames_paths = glob.glob("HDTF_train_processed/**/frames/*.jpg", recursive=True)
    # frames_paths = glob.glob("talking_face_others_train_processed/**/frames/*.jpg", recursive=True)
    print(f"# num frames: {len(frames_paths)}")

    ds = HDTFDataset(frames_paths)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=8)
    for sample in tqdm(dl):
        ref_frame, frame, _, mask, audio_feature = sample
        masked_frame = frame * mask

        mask = (
            (mask.permute(0, 2, 3, 1).squeeze(-1) * 255)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        frame = (
            (((frame.permute(0, 2, 3, 1) * 0.5) + 0.5) * 255)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        ref_frame = (
            (((ref_frame.permute(0, 2, 3, 1) * 0.5) + 0.5) * 255)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        masked_frame = (
            (((masked_frame.permute(0, 2, 3, 1) * 0.5) + 0.5) * 255)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        for i, (_mask, _frame, _ref_frame, _masked_frame) in enumerate(
            zip(mask, frame, ref_frame, masked_frame)
        ):
            Image.fromarray(_mask).save(f"mask-{i}.jpg")
            Image.fromarray(_frame).save(f"image-{i}.jpg")
            Image.fromarray(_ref_frame).save(f"ref_image-{i}.jpg")
            Image.fromarray(_masked_frame).save(f"masked_image-{i}.jpg")

        break
