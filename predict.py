# Prediction interface for Cog ⚙️
# https://cog.run/python

import glob
import os
import shutil
import tempfile

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from gfpgan.utils import GFPGANer

from tqdm.auto import tqdm

from knn import KNearestNeighbor
from musetalk.utils.blending import get_image
from musetalk.utils.preprocessing import (
    coord_placeholder,
    create_mask_from_2dlmk,
    extract_lower_face_landmarks,
    extract_mouth_landmarks,
    get_landmark_and_bbox,
    get_landmark_and_bbox_gx,
    musetalk_get_bbox_from_face_landmarks_and_bbox,
    detect_facial_landmarks
)
from musetalk.utils.utils import datagen, get_file_type, get_video_fps, load_all_model


def make_divisible(
    img: np.ndarray, divisor: int = 2 ** 5, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    """Resize an image (array) such that its sides are divisible by the divisor."""
    h, w = img.shape[:2]
    target_h = int(np.round(h / divisor)) * divisor
    target_w = int(np.round(w / divisor)) * divisor
    # print(f"h: {h}, w: {w}")
    # print(f"target_h: {target_h}, target_w: {target_w}")
    if target_h != h or target_w != w:
        img = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    return img


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels: int=6):
    """
    A and B values in range [0, 255] or [0, 1.0].
    mask values in range [0.0, 1.0]. 1.0 means take pixel value from A.
    """

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA = [
        gpA[num_levels - 1]
    ]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in range(num_levels - 1, 0, -1):
        # Laplacian: subtract upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        gm = gm[:, :, np.newaxis]
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_


def extract_video_frames(
    video_path: str, output_dir: str, max_frames: int = -1, output_ext: str = ".png"
):
    """Extract video frames to directory."""
    video = cv2.VideoCapture(video_path)
    i = 0
    output_paths = []
    while True:
        if max_frames != -1 and i >= max_frames:
            print(f"Max frames reached: {i}")
            break

        ret, frame = video.read()
        if not ret:
            break
        output_path = os.path.join(output_dir, f"{i:08d}{output_ext}")
        cv2.imwrite(output_path, frame)
        i += 1
        output_paths.append(output_path)
    video.release()
    return output_paths


def extract_audio(
    input_path, output_path, start_s: float = 0, duration_s: float = -1
) -> str:
    """Extract audio from input file."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    duration_str = ""
    if duration_s != -1:
        duration_str = f"-t {duration_s} "
    cmd = (
        f"ffmpeg -y -v fatal -ss {start_s} {duration_str}-i {input_path} {output_path}"
    )
    os.system(cmd)
    return output_path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # XXX GX Whether to enable the changes I made
        self.enable_gx_mods = True

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print("Loading audio processor, VAE, U-Net, positional encoding...")
        if self.enable_gx_mods:
            print("Loading models with GX mods...")
            audio_processor, vae, unet, pe = load_all_model(
                whisper_checkpoint="./models/whisper/tiny.pt",
                vae_model_id_or_path="./models/sd-vae-ft-mse/",
                unet_config_path="./musetalk-gx_mod.json",
                unet_checkpoint="./output/musetalk-finetune/checkpoint-464000/model.safetensors",
                audio_features_dim=5 * 384,
            )
        else:
            print("Loading models without GX mods...")
            audio_processor, vae, unet, pe = load_all_model()
        print("Done!")

        timesteps = torch.tensor([0], device=device)

        if device == "cuda":
            print("Putting models on half precision...")
            pe = pe.half()
            vae.vae = vae.vae.half()
            unet.model = unet.model.half()
            print("Done!")

        self.device = device
        self.audio_processor = audio_processor
        self.pe = pe
        self.vae = vae
        self.unet = unet
        self.timesteps = timesteps

        self.pe.eval()
        self.vae.eval()
        self.unet.eval()

        print("Loading GFPGAN...")
        self.face_enhancer = GFPGANer(
            # model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
            # model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
            # arch="RestoreFormer",
            upscale=1,
            device=device,
        )
        print("Done!")

    # def predict(
    #     self,
    #     video: Path = Input(description="video path"),
    #     audio: Path = Input(description="audio path"),
    #     bbox_shift: int = Input(description="BBox_shift value, px", default=0),
    #     cycle: bool = Input(
    #         description="Cycle video to smooth the first and last frames.",
    #         default=False,
    #     ),
    #     face_enhance: bool = Input(
    #         description="Enhance face with GFPGAN.", default=False
    #     ),
    # ) -> Path:
    def predict(
        self,
        video: str,
        audio: str,
        bbox_shift: int = 0,
        cycle: bool = False,
        face_enhance_strategy: str = "full_frame",  # no, cropped_face, full_frame, cropped_face+full_frame
        use_nn_ref_frame: bool = False,
    ) -> Path:
        """Run a single prediction on the model"""

        print(f"video: {video}")
        print(f"audio: {audio}")
        print(f"bbox_shift: {bbox_shift}")
        print(f"cycle: {cycle}")
        # print(f"face_enhance: {face_enhance}")
        print(f"face_enhance_strategy: {face_enhance_strategy}")
        print(f"use_nn_ref_frame: {use_nn_ref_frame}")

        video_path = str(video)
        audio_path = str(audio)

        audio_processor = self.audio_processor
        pe = self.pe
        vae = self.vae
        unet = self.unet
        timesteps = self.timesteps
        IMAGE_SIZE = (256, 256)

        # result_dir = tempfile.mkdtemp()
        result_dir = "debug"
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir, exist_ok=True)

        frames_dir = os.path.join(result_dir, "frames")
        cropped_frames_dir = os.path.join(result_dir, "cropped_frames")
        masked_frames_dir = os.path.join(result_dir, "masked_frames")
        output_frames_dir = os.path.join(result_dir, "output_frames")
        output_vid_name = os.path.join(result_dir, "output.mp4")

        #######################################################################
        # Extract video frames
        #######################################################################

        print("Extracting video frames...")

        if get_file_type(video_path) == "video":
            os.makedirs(frames_dir, exist_ok=True)

            # XXX GX For some reason the number of frames * fps doesn't match the duration!
            # cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
            # os.system(cmd)
            # input_img_list = sorted(
            #     glob.glob(os.path.join(save_dir_full, "*.[jpJP][pnPN]*[gG]"))
            # )

            # Read and save frames
            input_img_list = extract_video_frames(video_path, output_dir=frames_dir)

            fps = get_video_fps(video_path)
            input_type = "video"

        elif get_file_type(video_path) == "image":
            input_img_list = [
                video_path,
            ]
            fps = 25.0
            input_type = "image"

        elif os.path.isdir(video_path):  # input img folder
            input_img_list = glob.glob(os.path.join(video_path, "*.[jpJP][pnPN]*[gG]"))
            input_img_list = sorted(
                input_img_list,
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
            )
            fps = 25.0
            input_type = "video"

        else:
            raise ValueError(
                f"{video_path} should be a video file, an image file or a directory of images"
            )

        print(f"num video frames: {len(input_img_list)}")
        print(f"fps: {fps}")

        print("Done!")

        #######################################################################
        # Extract audio features
        #######################################################################

        print("Extracting audio features...")

        if use_nn_ref_frame:
            # Extract the original audio from the video
            print(f"Extracting audio from video file: {video_path}...")
            ori_audio_path = os.path.join(result_dir, "ori_audio.wav")
            ori_audio_path = extract_audio(video_path, ori_audio_path)

            # Extract audio features from the original audio
            if self.enable_gx_mods:
                whisper_feature_ori = audio_processor.audio2feat_gx(ori_audio_path)
            else:
                whisper_feature_ori = audio_processor.audio2feat(ori_audio_path)
            print(f"whisper_feature_ori.shape: {whisper_feature_ori.shape}")
            whisper_chunks_ori = audio_processor.feature2chunks(
                feature_array=whisper_feature_ori, fps=fps
            )
            print(
                f"whisper_chunks_ori.shape: {len(whisper_chunks_ori)}, whisper_chunks_ori[0].shape: {whisper_chunks_ori[0].shape}"
            )

        # Extract the audio from the audio video if necessary
        if audio_path.endswith(".mp4"):
            print(f"Extracting audio from audio file: {audio_path}...")
            _audio_path = os.path.join(result_dir, "audio.wav")
            audio_path = extract_audio(audio_path, _audio_path)

        # Extract audio features from the driving audio
        if self.enable_gx_mods:
            whisper_feature = audio_processor.audio2feat_gx(audio_path)
        else:
            whisper_feature = audio_processor.audio2feat(audio_path)
        print(f"whisper_feature.shape: {whisper_feature.shape}")
        whisper_chunks = audio_processor.feature2chunks(
            feature_array=whisper_feature, fps=fps
        )
        print(
            f"whisper_chunks.shape: {len(whisper_chunks)}, whisper_chunks[0].shape: {whisper_chunks[0].shape}"
        )

        print("Done!")

        #######################################################################
        # Face detection
        #######################################################################

        # print("Face detection...")

        landmark_list, bbox_list = get_landmark_and_bbox_gx(input_img_list)
        # print(f"landmark_list: {landmark_list}")
        # print(f"bbox_list: {bbox_list}")
        coord_list = musetalk_get_bbox_from_face_landmarks_and_bbox(
            keypoints=landmark_list, bbox_list=bbox_list
        )
        frame_list = input_img_list

        lower_face_landmarks = extract_lower_face_landmarks(
            facial_landmarks=landmark_list
        )
        # mouth_landmarks_list = extract_mouth_landmarks_from_facial_landmarks(
        #     facial_landmarks=landmark_list
        # )

        # coord_list, frame_list, mouth_landmarks_list = get_landmark_and_bbox(
        #     input_img_list, bbox_shift
        # )
        assert frame_list == input_img_list
        assert len(coord_list) == len(frame_list)
        assert len(frame_list) == len(lower_face_landmarks)

        # print("Done!")

        #######################################################################
        # Fit nearest neighbor search
        #######################################################################

        if use_nn_ref_frame:
            print("Fitting nearest neighbor face retriever...")

            os.makedirs(cropped_frames_dir, exist_ok=True)

            count = 0
            ref_indices = []
            for i, (bbox, frame, _lower_face_landmarks, whisper_chunk) in enumerate(
                zip(coord_list, frame_list, lower_face_landmarks, whisper_chunks_ori)
            ):
                if bbox == coord_placeholder:
                    # XXX GX No big deal here, we just don't use the current frame as a reference frame
                    continue

                frame = cv2.imread(frame)

                x1, y1, x2, y2 = bbox
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(
                    crop_frame, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4
                )
                cv2.imwrite(
                    os.path.join(cropped_frames_dir, f"{count:08d}.png"), crop_frame
                )

                ref_indices.append(i)
                count += 1

            assert len(ref_indices) == count

            # Fit nearest neighbor retriever
            whisper_chunks_ori = np.stack(whisper_chunks_ori)
            print(f"whisper_chunks_ori.shape: {whisper_chunks_ori.shape}")
            X = whisper_chunks_ori.reshape(whisper_chunks_ori.shape[0], -1)[
                np.array(ref_indices)
            ]
            print(f"X.shape: {X.shape}")
            knn = KNearestNeighbor(d=X.shape[-1])
            knn.fit(X)

            print("Done!")

        #######################################################################
        # Encode cropped face images to latents
        #
        # TODO:
        # 1. Use the lower face mask instead of lower frame mask. Also take note to erode/dilate the lower face mask so ask to give the model more hints to maintain details like facial hair.
        #######################################################################

        os.makedirs(masked_frames_dir, exist_ok=True)

        if input_type == "image":
            coord_list = coord_list * len(whisper_chunks)
            frame_list = frame_list * len(whisper_chunks)
            lower_face_landmarks = lower_face_landmarks * len(whisper_chunks)

        input_latent_list = []
        prev_i = -1
        for i, (bbox, frame, _lower_face_landmarks, whisper_chunk) in enumerate(
            zip(coord_list, frame_list, lower_face_landmarks, whisper_chunks)
        ):
            if bbox == coord_placeholder:
                # First frame or current frame to last valid frame greater than threshold
                if i == 0 or (i != 0 and i - prev_i > 5):
                    print(
                        f"Invalid bounding box: {bbox}, current frame: {i}, previous valid frame: {prev_i}"
                    )
                    continue

            prev_i = i

            frame = cv2.imread(frame)

            x1, y1, x2, y2 = bbox
            # print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(
                crop_frame, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4
            )

            # Construct lower face mask from facial landmarks
            lower_face_mask = create_mask_from_2dlmk(frame, _lower_face_landmarks)
            lower_face_mask = lower_face_mask[y1:y2, x1:x2]
            if lower_face_mask.ndim == 3:
                lower_face_mask = lower_face_mask[..., 0]
            lower_face_mask = cv2.resize(
                lower_face_mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST
            )

            # print(f"crop_mouth_mask - shape: {crop_mouth_mask.shape}, dtype: {crop_mouth_mask.dtype}, min: {crop_mouth_mask.min()}, max: {crop_mouth_mask.max()}")
            # XXX
            # lower_face_mask = np.zeros_like(crop_mouth_mask)
            # lower_face_mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            # lower_face_mask[144:, :] = 255  # XXX
            # lower_face_mask[128:,:16] = 0
            # lower_face_mask[128:,-16:] = 0
            # lower_face_mask[128+32:-32,32:-32] = 255

            inpaint_mask = lower_face_mask
            valid_mask = 255 - inpaint_mask

            # Dilate the valid mask, shrink the lower face mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            valid_mask = cv2.dilate(valid_mask, kernel, iterations=3)

            alpha = valid_mask/np.float32(255)
            if alpha.ndim == 2:
                alpha = alpha[..., None]
            masked_face = (crop_frame * alpha).astype(np.uint8)
            cv2.imwrite(os.path.join(masked_frames_dir, f"{i:08d}.jpg"), masked_face)

            ref_crop_frame = (
                None  # None meaning use the current frame as the reference frame
            )
            if use_nn_ref_frame:
                # Get nearest neighbor
                assert knn._fitted
                j = knn.predict(whisper_chunk.reshape(1, -1))[0][0]

                print(f"current frame: {i}, nearest neighbor frame: {j}")
                ref_crop_frame = cv2.imread(
                    os.path.join(cropped_frames_dir, f"{j:08d}.png")
                )

            if self.enable_gx_mods:
                latents = vae.get_latents_for_unet(
                    crop_frame,
                    mask=valid_mask,
                    ref_img=ref_crop_frame,
                    enable_gx_mods=True,
                )
            else:
                latents = vae.get_latents_for_unet(
                    crop_frame, mask=valid_mask, ref_img=ref_crop_frame
                )
            input_latent_list.append(latents)

        #######################################################################
        # Batched (latent inpainting) inference
        #######################################################################

        # print("Inference...")
        batch_size = 8

        # GX
        print(f"len(frame_list): {len(frame_list)}")
        print(f"len(whisper_chunks): {len(whisper_chunks)}")
        seq_len = min(len(frame_list), len(whisper_chunks))
        print(f"seq_len: {seq_len}")
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

        video_num = len(whisper_chunks)
        gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
        res_frame_list = []
        for i, (whisper_batch, latent_batch) in enumerate(
            tqdm(
                gen,
                total=int(np.ceil(float(video_num) / batch_size)),
                desc="Face inpainting",
            )
        ):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(
                device=unet.device, dtype=unet.model.dtype
            )  # torch, B, 5*N,384
            if self.enable_gx_mods:
                audio_feature_batch = audio_feature_batch.reshape(
                    audio_feature_batch.size(0), -1, 5 * 384
                )

            # XXX
            audio_feature_batch = audio_feature_batch[:, [4, 5],:]

            print(f"audio_feature_batch.shape: {audio_feature_batch.shape}")

            audio_feature_batch = pe(audio_feature_batch)

            latent_batch = latent_batch.to(dtype=unet.model.dtype)

            pred_latents = unet.model(
                latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)

        # print("Done!")

        #######################################################################
        # Paste inpainted image back to the original video
        #
        # TODO
        # 1. Extract only the mouth and paste it onto the original face.
        #######################################################################

        # print("Pasting inpainted image back to the original video...")

        os.makedirs(output_frames_dir, exist_ok=True)

        for i, res_frame in enumerate(
            tqdm(res_frame_list, desc="Pasting inpainted image back to original video")
        ):
            # XXX GX len(coord_list_cycle) == len(frame_list_cycle) could be greater than len(res_frame_list)
            # How to handle this?

            # bbox = coord_list_cycle[i % (len(coord_list_cycle))]
            # ori_frame = cv2.imread(frame_list_cycle[i % (len(frame_list_cycle))])

            ori_frame_index = i

            bbox = coord_list_cycle[ori_frame_index]
            ori_frame = cv2.imread(frame_list_cycle[ori_frame_index])

            x1, y1, x2, y2 = bbox

            # XXX How to deal with this? Should we just skip like how its originally done? Or should we take the last available frame?
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception as e:
                print(f"Invalid bounding box: {bbox}")
                continue

            # XXX GX Face restoration on the inpainted cropped face
            if face_enhance_strategy in [
                "cropped_face",
                "cropped_face+full_frame",
            ]:
                # XXX GX GFPGAN uses BGR image input: https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py#L131-L139, res_frame is BGR also so ok
                # print("face enhance before combining frame")
                _, _, res_frame = self.face_enhancer.enhance(
                    res_frame,
                    has_aligned=False,
                    only_center_face=True,
                    paste_back=True,
                    weight=0.5,
                )

            combined_frame = get_image(ori_frame, res_frame, bbox)

            # XXX GX Face restoration on the pasted back (full) frame
            if face_enhance_strategy in [
                "full_frame",
                "cropped_face+full_frame",
            ]:
                # print("face enhance after combining frame")
                _, _, combined_frame = self.face_enhancer.enhance(
                    combined_frame,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=0.5,
                )

            facial_landmarks = detect_facial_landmarks([combined_frame, ])
            mouth_landmarks = extract_mouth_landmarks(facial_landmarks)
            mouth_landmarks = mouth_landmarks[0]
            mouth_mask = create_mask_from_2dlmk(combined_frame, mouth_landmarks)
            if mouth_mask.ndim == 3:
                mouth_mask = mouth_mask[..., 0]

            # Dilate the valid mask, shrink the lower face mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mouth_mask = cv2.dilate(mouth_mask, kernel=kernel, iterations=15)

            # Take the mouth from the generated face image, other areas from the original face image and perform Gaussian Pyramid Blending
            pyramid_levels = 5
            ori_frame_div = make_divisible(ori_frame[y1:y2, x1:x2], divisor=2 ** pyramid_levels)  # [0, 255]
            combined_frame_div = make_divisible(combined_frame[y1:y2, x1:x2], divisor=2 ** pyramid_levels)  # [0, 255]
            mouth_mask_div = make_divisible(mouth_mask[y1:y2, x1:x2], divisor=2 ** pyramid_levels)
            mouth_mask_div = mouth_mask_div/np.float32(255)  # [0, 1]
            # print(f"ori_frame_div.shape: {ori_frame_div.shape}, dtype: {ori_frame_div.dtype}")
            # print(f"combined_frame_div.shape: {combined_frame_div.shape}, dtype: {combined_frame_div.dtype}")
            # print(f"mouth_mask_div.shape: {mouth_mask_div.shape}, dtype: {mouth_mask_div.dtype}")
            res_frame2 = Laplacian_Pyramid_Blending_with_mask(ori_frame_div, combined_frame_div, 1.0 - mouth_mask_div, num_levels=pyramid_levels)
            res_frame2 = res_frame2.astype(np.uint8)
            res_frame2 = cv2.resize(res_frame2, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

            combined_frame2 = get_image(ori_frame, res_frame2, bbox)

            cv2.imwrite(f"{output_frames_dir}/{str(i).zfill(8)}.png", combined_frame2)

        # print("Done!")

        #######################################################################
        # Image to video and combine video with audio
        #######################################################################

        cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {output_frames_dir}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
        print(cmd_img2video)
        os.system(cmd_img2video)

        # f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
        cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 -shortest {output_vid_name}"
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)

        return Path(output_vid_name)


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()

    # predictor.predict(
    #     video="VID-20240517-WA0007.mp4",
    #     audio="warren-bad.mp4",
    #     face_enhance_strategy="full_frame",
    # )
    # predictor.predict(
    #     video="VID-20240517-WA0007.mp4",
    #     audio="elon.wav",
    #     face_enhance_strategy="full_frame",
    # )
    # predictor.predict(
    #     video="VID-20240517-WA0007.mp4",
    #     audio="1.wav",
    #     face_enhance_strategy="full_frame",
    # )
    # predictor.predict(
    #     video="VID-20240517-WA0007.mp4",
    #     audio="2.wav",
    #     face_enhance_strategy="full_frame",
    # )    
    # predictor.predict(video="1.mp4", audio="1.wav", face_enhance_strategy="full_frame")
    # predictor.predict(video="1.mp4", audio="2.wav", face_enhance_strategy="full_frame")
    predictor.predict(video="1.mp4", audio="elon.wav", face_enhance_strategy="full_frame")
