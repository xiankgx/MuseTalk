import pickle

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from face_detection import FaceAlignment, LandmarksType
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples

# initialize the mmpose model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = (
    "./musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"
)
checkpoint_file = "./models/dwpose/dw-ll_ucoco_384.pth"
model = init_model(config_file, checkpoint_file, device=device)

# initialize the face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

# maker if the bbox is not sufficient
coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized


def read_imgs(img_list):
    frames = []
    # print('reading images...')
    # for img_path in tqdm(img_list):
    for img_path in img_list:
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def get_bbox_range(img_list, upperbondrange: int = 0, batch_size_fa: int = 1):
    frames = read_imgs(img_list)

    batches = [
        frames[i : i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)
    ]

    coords_list = []
    landmarks = []

    if upperbondrange != 0:
        print(
            "get key_landmark and face bounding boxes with the bbox_shift:",
            upperbondrange,
        )
    else:
        print("get key_landmark and face bounding boxes with the default value")

    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord = face_land_mark[
                29
            ]  # np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = (
                    upperbondrange + half_face_coord[1]
                )  # 手动调整  + 向下（偏29）  - 向上（偏28）

    text_range = f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range


def get_landmark_and_bbox(img_list, upperbondrange: int = 0, batch_size_fa: int = 1):
    # frames = read_imgs(img_list)

    # batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    batches = [
        img_list[i : i + batch_size_fa] for i in range(0, len(img_list), batch_size_fa)
    ]

    coords_list = []
    coords_list_mouth = []  # GX
    landmarks = []

    if upperbondrange != 0:
        print(
            "get key_landmark and face bounding boxes with the bbox_shift:",
            upperbondrange,
        )
    else:
        print("get key_landmark and face bounding boxes with the default value")

    average_range_minus = []
    average_range_plus = []
    # for fb in tqdm(batches):
    for fb in batches:
        fb = read_imgs(fb)  # XXX GX

        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)

        # GX
        mouth_land_mark = face_land_mark[-19:]
        # print(f"mouth_land_mark: {mouth_land_mark}")

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                coords_list_mouth.append(mouth_land_mark)
                continue

            half_face_coord = face_land_mark[
                29
            ]  # np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = (
                    upperbondrange + half_face_coord[1]
                )  # 手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            upper_bond = half_face_coord[1] - half_face_dist

            f_landmark = (
                np.min(face_land_mark[:, 0]),
                int(upper_bond),
                np.max(face_land_mark[:, 0]),
                np.max(face_land_mark[:, 1]),
            )
            x1, y1, x2, y2 = f_landmark

            if (
                y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0
            ):  # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w, h = f[2] - f[0], f[3] - f[1]
                print("error bbox:", f)
            else:
                coords_list += [f_landmark]

            coords_list_mouth.append(mouth_land_mark)

    print(
        "********************************************bbox_shift parameter adjustment**********************************************************"
    )
    print(
        f"Total frame:「{len(img_list)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    )
    print(
        "*************************************************************************************************************************************"
    )
    return coords_list, img_list, coords_list_mouth
    # return coords_list,frames,coords_list_mouth


class ImageDataset(Dataset):

    def __init__(self, image_paths, transform=None, read_img=cv2.imread):
        self.image_paths = image_paths
        self.transform = transform
        self.read_img = read_img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.read_img(image_path)
        return image


def get_landmark_and_bbox_gx(img_list: List[str], batch_size: int=4, num_workers: int=4):
    def collate_fn(samples):
        """Custom collate function to just stack the images instead of converting them to torch tensors."""
        return np.stack(samples, axis=0)

    ds = ImageDataset(image_paths=img_list)
    dl = DataLoader(ds, shuffle=False, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    keypoints_list = []
    bbox_list = []

    for batch in tqdm(dl, desc="Face landmarks and bounding box detection"):
        images = batch

        results = [merge_data_samples(inference_topdown(model, image)) for image in images]
        keypoints = [result.pred_instances.keypoints[0][23:91] for result in results]   # 23:91 for the 68 face landmarks
        keypoints = np.stack(keypoints, axis=0).astype(np.int32)  # b, 68, 2

        bbox = fa.get_detections_for_batch(images)

        keypoints_list.append(keypoints)
        bbox_list += bbox

    keypoints = np.concatenate(keypoints_list, axis=0)
    # Can't concat bbox_list because it may contain None 

    assert len(keypoints) == len(img_list)
    assert len(bbox_list) == len(img_list)

    return keypoints, bbox_list


def extract_mouth_landmarks_from_facial_landmarks(facial_landmarks: np.ndarray) -> np.ndarray:
    # last 19 landmarks in the face landmarks (49-68, 1-indexed)
    # b, 68, 2 -> b, 19, 2
    return facial_landmarks[:, -19:]


def extract_lower_face_landmarks(facial_landmarks: np.ndarray) -> np.ndarray:
    return facial_landmarks[:, list(range(32-1, 36+1-1)) + list(range(3-1, 15-1)) + list(range(32-1, 36-1))]


def musetalk_get_bbox_from_face_landmarks_and_bbox(keypoints: np.ndarray, bbox_list: List[Optional[Tuple[int]]], y_offset: int=0):
    """Replicating the original logic in musetalk."""

    if not len(keypoints) == len(bbox_list):
        raise Exception("Different keypoints list and bbox list lengths.")

    coords_list = []
    for face_land_mark, bbox in tqdm(zip(keypoints, bbox_list), desc="Determining bbox from facial landmarks and bbox"):
            if bbox is None:
                coords_list.append(coord_placeholder)

            # https://www.researchgate.net/profile/Fabrizio-Falchi/publication/338048224/figure/fig1/AS:837860722741255@1576772971540/68-facial-landmarks.jpg
            half_face_coord = face_land_mark[29]

            half_face_y = half_face_coord[1]
            if y_offset:
                half_face_y += y_offset

            # LOL, it seems that there are no landmarks for the top of the head, hence this method of doing things...

            # index 1 for y-coordinates, get the lowest face point and minus the center
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_y
            upper_bond = half_face_y - half_face_dist

            f_landmark = (
                np.min(face_land_mark[:, 0]),
                int(upper_bond),
                np.max(face_land_mark[:, 0]),
                np.max(face_land_mark[:, 1]),
            )
            x1, y1, x2, y2 = f_landmark

            if (
                y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0
            ):  # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
            else:
                coords_list += [f_landmark]
    assert len(coords_list) == len(keypoints)
    return coords_list



if __name__ == "__main__":
    img_list = [
        "./results/lyria/00000.png",
        "./results/lyria/00001.png",
        "./results/lyria/00002.png",
        "./results/lyria/00003.png",
    ]
    crop_coord_path = "./coord_face.pkl"
    coords_list, full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, "wb") as f:
        pickle.dump(coords_list, f)

    for bbox, frame in zip(coords_list, full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print("Cropped shape", crop_frame.shape)

        # cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)
