import numpy as np
import torch
from tqdm import tqdm
import cv2
import numpy as np

def save_video(video_clip, output_filename, fps=30):
    """
    Saves the video clip as a .mp4 file.

    Parameters:
    - video_clip: A NumPy array of shape [batch_size, channels, timestamps, height, width].
    - output_filename: The name of the output video file.
    - fps: Frames per second for the output video.
    """
    # Extract the shape of the video clip
    batch_size, channels, timestamps, height, width = video_clip.shape
    video_clip = video_clip.cpu().numpy()
    # Convert to the proper shape for OpenCV (timestamps, height, width, channels)
    video_clip = video_clip.squeeze(0)  # Assuming batch_size = 1
    video_clip = video_clip.transpose(1, 0, 2, 3)  # Change shape to [timestamps, channels, height, width]
    
    print("video_clip.shape", video_clip.shape)
    
    # Initialize a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    # Iterate over each timestamp/frame and write to the video file
    for frame in video_clip:
        # Convert the frame from (channels, height, width) to (height, width, channels)
        frame = np.transpose(frame, (1, 2, 0))  # Change shape to [height, width, channels]
        
        print("frame.shape", frame.shape)
        
        # Convert to uint8 for video writing
        frame = np.uint8(frame * 255)  # Assuming pixel values are between 0 and 1
        print(frame.max(), frame.min())
        # Write the frame to the output video file
        out.write(frame)
    
    # Release the VideoWriter object to finalize the video file
    out.release()
    print(f"Video saved as {output_filename}")


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)
    return x

def calculate_fvd(videos1, videos2, device, i3d, method='styleganv'):
    print(method)
    if method == 'styleganv':
        from .fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from .fvd.videogpt.fvd import load_i3d_pretrained
        from .fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from .fvd.videogpt.fvd import frechet_distance

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)
    
    # print(videos1.shape, videos2.shape)
    
    fvd_results = {}

    # for calculate FVD, each clip_timestamp must >= 10
    # videos_clip [batch_size, channel, timestamps[:clip], h, w]
    videos_clip1 = videos1[:, :, :]
    videos_clip2 = videos2[:, :, :]
    # get FVD features
    feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
    feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
    
    # calculate FVD when timestamps[:clip]
    fvd_results[99] = frechet_distance(feats1, feats2)
    
    
    # if frechet_distance(feats1, feats2) > 400:
    #     save_video(videos_clip1, f"output_video1_{clip_timestamp}.mp4")
    #     save_video(videos_clip2, f"output_video2_{clip_timestamp}.mp4")
    #     exit(1)
        
    print(fvd_results)
    result = {
        "value": fvd_results,
        "video_setting": videos1.shape,
        "video_setting_name": "batch_size, channel, time, heigth, width",
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    import json
    result = calculate_fvd(videos1, videos2, device, method='videogpt')
    print(json.dumps(result, indent=4))

    result = calculate_fvd(videos1, videos2, device, method='styleganv')
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
