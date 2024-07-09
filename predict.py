# Prediction interface for Cog ⚙️
# https://cog.run/python

import copy
import glob
import os
import pickle
import shutil
import tempfile
from tqdm.auto import tqdm

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from omegaconf import OmegaConf

from musetalk.utils.blending import get_image
from musetalk.utils.preprocessing import (
    coord_placeholder,
    get_landmark_and_bbox,
    read_imgs,
)
from musetalk.utils.utils import datagen, get_file_type, get_video_fps, load_all_model


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        audio_processor, vae, unet, pe = load_all_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        timesteps = torch.tensor([0], device=device)

        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

        self.device = device
        self.audio_processor = audio_processor
        self.pe = pe
        self.vae = vae
        self.unet = unet
        self.timesteps = timesteps

    def predict(
        self,
        video: Path = Input(description="video path"),
        audio: Path = Input(description="audio path"),
        bbox_shift: int = Input(description="BBox_shift value, px", default=0),
        cycle: bool = Input(description="Cycle video to smooth the first and last frames.", default=False)
    ) -> Path:
        """Run a single prediction on the model"""

        video_path = str(video)
        audio_path = str(audio)

        audio_processor = self.audio_processor
        pe = self.pe
        vae = self.vae
        unet = self.unet
        timesteps = self.timesteps

        input_basename = os.path.basename(video_path).split(".")[0]
        audio_basename = os.path.basename(audio_path).split(".")[0]
        output_basename = f"{input_basename}_{audio_basename}"

        result_dir = tempfile.mkdtemp()

        result_img_save_path = os.path.join(
            result_dir, output_basename
        )  # related to video & audio inputs
        crop_coord_save_path = os.path.join(
            result_img_save_path, input_basename + ".pkl"
        )  # only related to video input
        os.makedirs(result_img_save_path, exist_ok=True)

        output_vid_name = os.path.join(result_dir, "output.mp4")

        ############################################## extract frames from source video ##############################################
        if get_file_type(video_path) == "video":
            save_dir_full = os.path.join(result_dir, input_basename)
            os.makedirs(save_dir_full, exist_ok=True)
            cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
            os.system(cmd)
            input_img_list = sorted(
                glob.glob(os.path.join(save_dir_full, "*.[jpJP][pnPN]*[gG]"))
            )
            fps = get_video_fps(video_path)
            print(f"fps: {fps}")
        elif get_file_type(video_path) == "image":
            input_img_list = [
                video_path,
            ]
            fps = 25.0
        elif os.path.isdir(video_path):  # input img folder
            input_img_list = glob.glob(os.path.join(video_path, "*.[jpJP][pnPN]*[gG]"))
            input_img_list = sorted(
                input_img_list,
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
            )
            fps = 25.0
        else:
            raise ValueError(
                f"{video_path} should be a video file, an image file or a directory of images"
            )

        # print(input_img_list)
        ############################################## extract audio feature ##############################################
        whisper_feature = audio_processor.audio2feat(audio_path)
        print(f"whisper_feature.shape: {whisper_feature.shape}")
        whisper_chunks = audio_processor.feature2chunks(
            feature_array=whisper_feature, fps=fps
        )
        print(f"whisper_chunks.shape: {len(whisper_chunks)}, whisper_chunks[0].shape: {whisper_chunks[0].shape}")

        ############################################## preprocess input image  ##############################################
        # if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        if os.path.exists(crop_coord_save_path):
            print("using extracted coordinates")
            with open(crop_coord_save_path, "rb") as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            print("extracting landmarks...time consuming")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
            with open(crop_coord_save_path, "wb") as f:
                pickle.dump(coord_list, f)

        i = 0
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(
                crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
            )
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)

        # GX
        seq_len = min(len(frame_list), len(whisper_chunks))
        frame_list = frame_list[:seq_len]
        coord_list = coord_list[:seq_len]
        input_latent_list = input_latent_list[:seq_len]
        whisper_chunks = whisper_chunks[:seq_len]

        # to smooth the first and the last frame
        if cycle:
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        else:
            frame_list_cycle = frame_list
            coord_list_cycle = coord_list
            input_latent_list_cycle = input_latent_list

        ############################################## inference batch by batch ##############################################
        print("start inference")
        video_num = len(whisper_chunks)
        batch_size = 8

        # assert len(whisper_chunks) == len(input_latent_list_cycle), f"len(whisper_chunks): {len(whisper_chunks)}, len(input_latent_list_cycle): {len(input_latent_list_cycle)}"
        gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
        res_frame_list = []
        for i, (whisper_batch, latent_batch) in enumerate(
            tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))
        ):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(
                device=unet.device, dtype=unet.model.dtype
            )  # torch, B, 5*N,384
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)

            pred_latents = unet.model(
                latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)

        ############################################## pad to full image ##############################################
        print("pad talking image to original video")
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i % (len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i % (len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception as e:
                # print(bbox)
                print(f"Exception: {str(e)}")
                continue

            combine_frame = get_image(ori_frame, res_frame, bbox)
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)


        cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
        print(cmd_img2video)
        os.system(cmd_img2video)

        # f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
        cmd_combine_audio = (
            f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 -shortest {output_vid_name}"
        )
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)

        # os.remove("temp.mp4")
        # shutil.rmtree(result_img_save_path)
        # print(f"result is save to {output_vid_name}")

        return Path(output_vid_name)
