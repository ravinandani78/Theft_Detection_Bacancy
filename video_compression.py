import subprocess
import os

def compress_4k_video(input_path, output_path, downscale=False):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    command = [
        'ffmpeg',
        '-i', input_path,
        '-vcodec', 'libx265',
        '-crf', '28',
        '-preset', 'ultrafast',
        '-acodec', 'aac',
        '-b:a', '128k'
    ]

    if downscale:
        command += ['-vf', 'scale=1920:1080']

    command.append(output_path)

    print("Running command:", " ".join(command))

    subprocess.run(command, check=True)

compress_4k_video(
    input_path='IMG_7274.MOV',
    output_path='output_7276.mp4',
    downscale=False  
)