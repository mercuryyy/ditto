#!/usr/bin/env python3
import json
import base64
import sys
import os

def convert_json_to_video(json_file_path, output_video_path=None):
    """
    Convert base64 video data from JSON file to a video file.
    
    Args:
        json_file_path (str): Path to the JSON file containing video data
        output_video_path (str): Path for the output video file (optional)
    """
    
    # Read the JSON file
    print(f"Reading JSON file: {json_file_path}")
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return False
    
    # Extract the base64 video data
    try:
        video_base64 = data['output']['video_base64']
        print(f"Found base64 video data (length: {len(video_base64)} characters)")
    except KeyError as e:
        print(f"Error: Expected key not found in JSON - {e}")
        print("Expected structure: {'output': {'video_base64': '...'}}")
        return False
    
    # Decode the base64 data
    print("Decoding base64 data...")
    try:
        video_data = base64.b64decode(video_base64)
        print(f"Successfully decoded {len(video_data)} bytes of video data")
    except Exception as e:
        print(f"Error decoding base64 data: {e}")
        return False
    
    # Determine output file path
    if output_video_path is None:
        # Generate output filename based on input filename
        base_name = os.path.splitext(json_file_path)[0]
        output_video_path = f"{base_name}_output.mp4"
    
    # Write the video data to file
    print(f"Writing video file: {output_video_path}")
    try:
        with open(output_video_path, 'wb') as f:
            f.write(video_data)
        print(f"Successfully created video file: {output_video_path}")
        
        # Display file size
        file_size = os.path.getsize(output_video_path)
        print(f"Output file size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        return True
    except Exception as e:
        print(f"Error writing video file: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_video.py <json_file> [output_video_file]")
        print("Example: python convert_video.py video.json")
        print("Example: python convert_video.py video.json my_video.mp4")
        return
    
    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_json_to_video(json_file, output_file)
    
    if success:
        print("✅ Video conversion completed successfully!")
    else:
        print("❌ Video conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
