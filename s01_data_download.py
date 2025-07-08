import os
import shutil
import gdown
import argparse
import subprocess
import concurrent.futures

wildtrack_video_links = [
    "https://drive.google.com/uc?id=1sGUnExmJM2_tFuBd9LNlexf0LN2m0_c-",  # Camera 1
    "https://drive.google.com/uc?id=1OnaRN2qYhZ2n4rSaNZQJzQPd1Cl2fluk",  # Camera 2
    "https://drive.google.com/uc?id=1I7ARLVVfZqdZTbYG_TPEfDt1Bc8N35Lb",  # Camera 3
    "https://drive.google.com/uc?id=1sXn70X-bV_YGPv43r4-iMtK_Js09eUVB",  # Camera 4
    "https://drive.google.com/uc?id=1ExTyezMWqmLefi2kJTQwYxZivJuag7G5",  # Camera 5
    "https://drive.google.com/uc?id=1NM4kNjdyiC6JioOT90s9vFmMUkeSnV0z",  # Camera 6
    "https://drive.google.com/uc?id=1pZ3pWBuaLgPWGfOcY-tAtsBAUyQZW_YX",  # Camera 7
]

def download_video(link, output):
    '''
    Downloads the video from the specified link to the output path.
    
    :param link: Link to the video file
    :param output: Path to save the downloaded video
    '''
    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))
    gdown.download(link, output, quiet=False)



def cut_video(input_video, output_video, duration):
    """
    Cuts the video to the specified duration.
    
    :param input_video: Path to the original video
    :param output_video: Path to the output cut video
    :param duration: Duration to keep in the video (in seconds)
    """
    command = [
        'ffmpeg', '-i', input_video, '-t', str(duration), '-c', 'copy', output_video
    ]
    subprocess.run(command)

def extract_frames(video_path, output_folder, fps):
    """
    Extracts frames from the video at the specified frame rate.

    :param video_path: Path to the video
    :param output_folder: Directory to save the extracted frames
    :param fps: Frames per second to extract
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    command = [
        'ffmpeg', '-i', video_path, '-vf', f'fps={fps}', 
        os.path.join(output_folder, '%08d.png')
    ]
    subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wildtrack video data download.')
    parser.add_argument('--official-data-dir', type=str, default='/home/jiahao/Downloads/data/Wildtrack', help='Directory to the official dataset.')
    parser.add_argument('--data-dir', type=str, default='/home/jiahao/Downloads/data/wildtrack', help='Directory to save the data.')
    parser.add_argument('--duration', type=int, default=35, help='Cut the first N minutes of the video.')
    parser.add_argument('--fps', type=int, default=2, help='The number of frames to extract per second.')
    parser.add_argument('--output-folder', type=str, default='Image_subsets', help='The folder to save the extracted frames.')
    
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # Step 1: Download the videos
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, link in enumerate(wildtrack_video_links):
            if os.path.exists(os.path.join(args.data_dir, 'videos', f'Camera{i+1}.mp4')):
                continue
            output = os.path.join(args.data_dir, 'videos', f'Camera{i+1}.mp4')
            futures.append(executor.submit(download_video, link, output))
        
        # Wait for all the threads to complete
        concurrent.futures.wait(futures)

    for cam_idx, input_video in enumerate(sorted(os.listdir(os.path.join(args.data_dir, 'videos')))):
        if 'cut' in input_video: 
            continue # ignore the temporary cutted videos
        # Step 2: Cut the video to the first 15 minutes 
        input_video = os.path.join(args.data_dir, 'videos', input_video)
        dir_name = os.path.dirname(input_video)
        base_name = os.path.basename(input_video).split('.')[0]
        # if base_name in ['Camera1', 'Camera2']:
        #     continue
        cut_video_output = os.path.join(dir_name, f'{base_name}_cut.mp4')
        duration = args.duration * 60
        cut_video(input_video, cut_video_output, duration)
        
        # Step 3: Extract 10 frames per second from the cut video
        output_folder = os.path.join(args.data_dir, args.output_folder, f"C{cam_idx+1}")
        extract_frames(cut_video_output, output_folder, args.fps)
        
        # Step 4: Delete the cutted videos
        os.remove(cut_video_output)
    
        # Step 5: Reorganize the extracted frames into the correct folder structure
        for frame in sorted(os.listdir(output_folder)):
            frame_path = os.path.join(output_folder, frame)
            new_frame_path = os.path.join(args.data_dir, args.output_folder, frame.split('.')[0], f"cam{cam_idx+1}.png")
            # copy the frame to the new location and rename it
            if not os.path.exists(os.path.dirname(new_frame_path)):
                os.makedirs(os.path.dirname(new_frame_path))
            os.rename(frame_path, new_frame_path)
            
        # Step 6: Remove the empty folders
        # os.remove(output_folder)
        shutil.rmtree(output_folder) 
        
    # Step 7: prepare calibrations for the wildtrack dataset
    # calibrations_folder = args.official_data_dir # The path to Wildtrack dataset calibrations
    # # copy the calibration files to the correct location `--data-dir`
    # shutil.copytree(calibrations_folder, os.path.join(args.data_dir, 'calibrations'), dirs_exist_ok=True)
            
            
            