import glob
import os
import pickle
import shutil

import cv2
from tqdm.auto import tqdm

# from musetalk.utils.preprocessing import coord_placeholder  # read_imgs,
from musetalk.utils.preprocessing import get_landmark_and_bbox, get_landmark_and_bbox_gx, musetalk_get_bbox_from_face_landmarks_and_bbox
from musetalk.utils.utils import get_video_fps
from musetalk.whisper.audio2feature import Audio2Feature


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


def process(
    video_path, outdir, audio_processor, bbox_shift: int = 0, audio_feat_length=[2, 2]
) -> None:
    try:
        # Extract audio from video
        audio_path = video_path

        video_id = os.path.splitext(os.path.basename(video_path))[0]
        _outdir = os.path.join(outdir, video_id)
        os.makedirs(_outdir, exist_ok=True)

        if os.path.isfile(os.path.join(_outdir, "facial_landmarks.pkl")):
            return

        fps = get_video_fps(video_path)

        frames_save_path = os.path.join(_outdir, "frames")
        os.makedirs(frames_save_path, exist_ok=True)

        audio_features_save_path = os.path.join(_outdir, "audio_features")
        os.makedirs(audio_features_save_path, exist_ok=True)

        # Encode audio
        # np.ndarray
        if os.path.isfile(os.path.join(_outdir, "audio_features.pkl")):
            print("Using existing audio features...")
            with open(os.path.join(_outdir, "audio_features.pkl"), "rb") as f:
                whisper_chunks = pickle.load(f)
        else:
            whisper_feature = audio_processor.audio2feat_gx(audio_path)
            whisper_chunks = audio_processor.feature2chunks(
                feature_array=whisper_feature, fps=fps, audio_feat_length=audio_feat_length
            )

        if True:
            # XXX Temporary disabled
            # Extract video frames
            # video = cv2.VideoCapture(video_path)
            # i = 0
            # while True:
            #     ret, frame = video.read()
            #     if not ret:
            #         break
            #     cv2.imwrite(os.path.join(frames_save_path, f"{i:08d}.jpg"), frame)
            #     i += 1
            # video.release()
            input_img_list = sorted(
                glob.glob(os.path.join(frames_save_path, "*.[jpJP][pnPN]*[gG]"))
            )

            if abs(len(whisper_chunks) - len(input_img_list)) > 5:
                print(
                    f"diff number of frames: len(input_img_list): {len(input_img_list)}, len(whisper_chunks): {len(whisper_chunks)}"
                )
                shutil.rmtree(_outdir)
                return

            # Obtain bounding box coordinates, frames, and mouth landmarks
            coord_list, frame_list, landmarks = get_landmark_and_bbox(
                input_img_list, bbox_shift
            )

            # landmarks, bbox_list = get_landmark_and_bbox_gx(input_img_list, num_workers=0)
            # # print(f"landmark_list: {landmark_list}")
            # # print(f"bbox_list: {bbox_list}")
            # coord_list = musetalk_get_bbox_from_face_landmarks_and_bbox(
            #     keypoints=landmarks, bbox_list=bbox_list
            # )
            # frame_list = input_img_list

            assert len(coord_list) == len(frame_list)
            assert len(landmarks) == len(frame_list)
            # assert len(mouth_landmarks_list) == len(frame_list)

            # Save data

            # with open(os.path.join(_outdir, "audio_features.pkl"), "wb") as f:
            #     pickle.dump(whisper_chunks, f)

            # with open(os.path.join(_outdir, "coord_list.pkl"), "wb") as f:
            #     pickle.dump(coord_list, f)

            # with open(os.path.join(_outdir, "mouth_landmarks_list.pkl"), "wb") as f:
            #     pickle.dump(mouth_landmarks_list, f)

            with open(os.path.join(_outdir, "facial_landmarks.pkl"), "wb") as f:
                pickle.dump(landmarks, f)

            # for i, feature in enumerate(whisper_chunks):
            #     with open(os.path.join(audio_features_save_path, f"{i:08d}.pkl"), "wb") as f:
            #         pickle.dump(feature, f)

            # with open(os.path.join(audio_features_save_path, f"count.pkl"), "wb") as f:
            #     pickle.dump(len(whisper_chunks), f)
    except Exception as e:
        print(str(e))





if __name__ == "__main__":
    import argparse
    from functools import partial

    from torch.multiprocessing import Pool, Process, set_start_method

    try:
        set_start_method("spawn")
    except Exception as e:
        print(str(e))

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="HDTF")
    parser.add_argument("--output_dir", type=str, default="HDTF_train_processed")
    # parser.add_argument("--output_dir", type=str, default="talking_face_others_train_processed")
    parser.add_argument(
        "--audio_model_path", type=str, default="models/whisper/tiny.pt"
    )
    args = parser.parse_args()

    video_files = glob.glob(f"{args.input_dir}/*.mp4")
    # video_files = ["VID-20240517-WA0007.mp4", ]
    print(f"# video files: {len(video_files)}")

    audio_processor = Audio2Feature(model_path=args.audio_model_path)

    with Pool(processes=4) as pool:
        for i in tqdm(
            pool.imap_unordered(
                partial(process, outdir=args.output_dir, audio_processor=audio_processor),
                video_files,
            ),
            total=len(video_files),
        ):
            pass
