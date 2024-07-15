import glob
import os
import pickle

import cv2
import numpy as np
from torch.utils.data import Dataset

from musetalk.utils.preprocessing import (
    coord_placeholder,
    get_landmark_and_bbox,
    read_imgs,
)
from musetalk.utils.utils import get_video_fps


def read_frames(video_path):
    frames = []
    video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames


class HDTFDataset(Dataset):
    def __init__(self, video_paths, audio_processor, bbox_shift: int = 0):
        self.video_paths = video_paths
        self.bbox_shift = bbox_shift
        self.audio_processor = audio_processor

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        audio_path = self.video_paths[idx]

        cache_dir = os.path.join(
            os.path.dirname(video_path),
            ".cache",
            os.path.splitext(os.path.basename(video_path))[0],
        )
        os.makedirs(cache_dir, exist_ok=True)

        frames_save_path = os.path.join(cache_dir, "frames")
        if not os.path.isdir(frames_save_path):
            os.makedirs(frames_save_path)

            video = cv2.VideoCapture(video_path)
            i = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(frames_save_path, f"{i:08d}.png"), frame)
                i += 1
            video.release()

        input_img_list = sorted(
            glob.glob(os.path.join(frames_save_path, "*.[jpJP][pnPN]*[gG]"))
        )
        fps = get_video_fps(video_path)

        # Get crop coordinates, use cache if available
        crop_coord_save_path = os.path.join(cache_dir, "coord_list.pkl")
        if os.path.isfile(crop_coord_save_path):
            with open(crop_coord_save_path, "rb") as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            # print(f"input_img_list: {input_img_list}")
            coord_list, frame_list = get_landmark_and_bbox(
                input_img_list, self.bbox_shift
            )
            with open(crop_coord_save_path, "wb") as f:
                pickle.dump(coord_list, f)
        assert len(coord_list) == len(frame_list)

        crop_frames_save_path = os.path.join(cache_dir, "cropped_frames")
        if not os.path.isdir(crop_frames_save_path):
            os.makedirs(crop_frames_save_path)

            for i, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
                if bbox == coord_placeholder:
                    print("FAIL TO DETECT FACE, REPLACING WITH ALL ZERO IMAGE")
                    # continue
                    crop_frame = np.zeros((256, 256, 3), np.uint8)
                else:
                    x1, y1, x2, y2 = bbox
                    crop_frame = frame[y1:y2, x1:x2]

                crop_frame = cv2.resize(
                    crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
                )
                cv2.imwrite(
                    os.path.join(crop_frames_save_path, f"{i:08d}.png"), crop_frame
                )
        cropped_frames = sorted(glob.glob(f"{crop_frames_save_path}/*.*"))
        assert len(cropped_frames) == len(frame_list)

        whisper_feature_save_path = os.path.join(cache_dir, "audio_features.pkl")
        if os.path.isfile(whisper_feature_save_path):
            with open(whisper_feature_save_path, "rb") as f:
                whisper_chunks = pickle.load(f)
        else:
            # np.ndarray
            whisper_feature = self.audio_processor.audio2feat(audio_path)
            print(f"whisper_feature.shape: {whisper_feature.shape}")
            # list of np.ndarray
            whisper_chunks = self.audio_processor.feature2chunks(
                feature_array=whisper_feature, fps=fps
            )
            print(f"whisper_chunks.shape: {len(whisper_chunks)}")
            print(f"whisper_chunks[0].shape: {whisper_chunks[0].shape}")
            with open(whisper_feature_save_path, "wb") as f:
                pickle.dump(whisper_chunks, f)

        return {
            "video_path": video_path,
            # "audio_path": audio_path,
            "fps": fps,
            "num_frames": len(frame_list),
            "audio_feature": whisper_chunks
            # "num_audio_frames": len(whisper_chunks)
        }


if __name__ == "__main__":
    from musetalk.whisper.audio2feature import Audio2Feature

    audio_processor = Audio2Feature(model_path="models/whisper/tiny.pt")
    ds = HDTFDataset(
        video_paths=[
            "VID-20240517-WA0007.mp4",
        ],
        audio_processor=audio_processor,
    )
    sample = ds[0]
    print(sample)
