#!/usr/bin/env python3
"""
Script to create an animated GIF from PNG frames showing regression tree growth.
"""

import os
from PIL import Image
import glob

def create_gif_from_frames(frames_dir="gif", output_name="tree_stopping_animation.gif", duration=1000):
    """
    Create an animated GIF from PNG frames.
    
    Args:
        frames_dir: Directory containing the PNG frames
        output_name: Name of the output GIF file
        duration: Duration of each frame in milliseconds
    """
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frames_path = os.path.join(script_dir, frames_dir)
    
    # Get all PNG files and sort them
    frame_files = glob.glob(os.path.join(frames_path, "frame_*.png"))
    frame_files.sort()
    
    if not frame_files:
        print(f"No PNG frames found in {frames_path}")
        return
    
    print(f"Found {len(frame_files)} frames:")
    for f in frame_files:
        print(f"  {os.path.basename(f)}")
    
    # Load images
    images = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        images.append(img)
    
    # Save as GIF
    output_path = os.path.join(script_dir, output_name)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  # 0 means infinite loop
    )
    
    print(f"GIF created successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    create_gif_from_frames() 