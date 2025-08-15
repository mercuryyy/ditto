"""
Streaming Web UI for Ditto Talking Head using RunPod Serverless Endpoint
Supports progressive video loading for faster user experience
"""
import os
import base64
import requests
import gradio as gr
from PIL import Image
import io
import tempfile
import time
import threading
import json


class DittoStreamingWebUI:
    def __init__(self, runpod_endpoint, runpod_api_key):
        """
        Initialize the Streaming Web UI
        """
        self.runpod_endpoint = runpod_endpoint
        self.headers = {
            "Authorization": f"Bearer {runpod_api_key}",
            "Content-Type": "application/json"
        }
    
    def encode_image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def process_video_streaming(self, text, image, use_pytorch=False, enable_streaming=True):
        """
        Process video with streaming support
        """
        if not text:
            gr.Warning("Please enter some text to synthesize.")
            return None, "Please enter some text to synthesize."
        
        if image is None:
            gr.Warning("Please upload an image.")
            return None, "Please upload an image."
        
        try:
            # Prepare the request payload
            image_base64 = self.encode_image_to_base64(image)
            
            payload = {
                "input": {
                    "text": text,
                    "image_base64": image_base64,
                    "use_pytorch": use_pytorch,
                    "streaming_mode": enable_streaming
                }
            }
            
            # Send request to RunPod endpoint
            status_msg = "🚀 Sending request to RunPod..."
            gr.Info(status_msg)
            
            response = requests.post(
                f"{self.runpod_endpoint}/run",
                json=payload,
                headers=self.headers,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code != 200:
                error_msg = f"Failed to connect to RunPod endpoint. Status: {response.status_code}"
                gr.Error(error_msg)
                return None, error_msg
            
            result = response.json()
            
            # Check if the job was queued (async processing)
            if "id" in result:
                job_id = result["id"]
                status_msg = f"📋 Job queued with ID: {job_id}. Processing..."
                gr.Info(status_msg)
                return self.poll_job_status_streaming(job_id, enable_streaming)
            
            # Direct response (sync processing)
            if "output" in result:
                return self.process_output_streaming(result["output"], enable_streaming)
            
            error_msg = "Unexpected response format from RunPod"
            gr.Error(error_msg)
            return None, error_msg
            
        except requests.exceptions.Timeout:
            error_msg = "Request timed out. The video generation is taking too long."
            gr.Error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            gr.Error(error_msg)
            return None, error_msg
    
    def poll_job_status_streaming(self, job_id, enable_streaming=True, max_attempts=60):
        """Poll RunPod for job completion with streaming updates"""
        
        last_progress = 0
        status_messages = []
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.runpod_endpoint}/status/{job_id}",
                    headers=self.headers
                )
                
                if response.status_code != 200:
                    error_msg = f"Failed to check job status. Status: {response.status_code}"
                    gr.Error(error_msg)
                    return None, error_msg
                
                result = response.json()
                status = result.get("status")
                
                if status == "COMPLETED":
                    gr.Info("✅ Job completed successfully!")
                    status_messages.append("✅ Video generation completed!")
                    return self.process_output_streaming(result.get("output", {}), enable_streaming)
                    
                elif status == "FAILED":
                    error_msg = result.get("error", "Unknown error")
                    gr.Error(f"❌ Job failed: {error_msg}")
                    return None, f"Job failed: {error_msg}"
                    
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    # Show progress updates
                    progress_info = ""
                    if enable_streaming and "output" in result:
                        output = result["output"]
                        if "progress_updates" in output and output["progress_updates"]:
                            latest_progress = output["progress_updates"][-1]
                            current_progress = latest_progress.get("progress", 0)
                            
                            if current_progress > last_progress:
                                last_progress = current_progress
                                progress_info = f" (Progress: {current_progress:.1f}%)"
                    
                    if attempt % 3 == 0:  # Update every 15 seconds
                        status_msg = f"⚡ Job {status.lower().replace('_', ' ')}{progress_info}... (attempt {attempt+1}/{max_attempts})"
                        gr.Info(status_msg)
                        status_messages.append(status_msg)
                    
                    time.sleep(5)  # Wait 5 seconds before checking again
                    
                else:
                    error_msg = f"Unknown job status: {status}"
                    gr.Error(error_msg)
                    return None, error_msg
                    
            except Exception as e:
                error_msg = f"Error checking job status: {str(e)}"
                gr.Error(error_msg)
                return None, error_msg
        
        error_msg = "Job timed out after 5 minutes"
        gr.Error(error_msg)
        return None, error_msg
    
    def process_output_streaming(self, output, enable_streaming=True):
        """Process the output with streaming information"""
        if "error" in output:
            error_msg = f"Error from RunPod: {output['error']}"
            gr.Error(error_msg)
            return None, error_msg
        
        if "video_base64" not in output:
            error_msg = "No video data in response"
            gr.Error(error_msg)
            return None, error_msg
        
        try:
            # Decode the video from base64
            video_data = base64.b64decode(output["video_base64"])
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_file.write(video_data)
                
                # Generate status message with streaming info
                status_msg = "🎬 Video generated successfully!"
                
                if enable_streaming and output.get("streaming", False):
                    progress_updates = output.get("progress_updates", [])
                    if progress_updates:
                        total_frames = progress_updates[-1].get("total", 0) if progress_updates else 0
                        status_msg += f" (Streaming mode: {total_frames} frames processed)"
                    else:
                        status_msg += " (Streaming mode enabled)"
                else:
                    status_msg += " (Standard mode)"
                
                gr.Info(status_msg)
                return tmp_file.name, status_msg
                
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            gr.Error(error_msg)
            return None, error_msg
    
    def launch(self):
        """Launch the Gradio interface with streaming support"""
        with gr.Blocks(title="Ditto Talking Head Streaming Web UI", theme=gr.themes.Soft()) as interface:
            gr.Markdown(
                """
                # 🎭 Ditto Talking Head Streaming Web UI
                
                Generate lip-synced talking head videos from text and images using the Ditto model deployed on RunPod.
                Now with **streaming support** for faster video preview!
                
                ## How to use:
                1. Enter the text you want to synthesize
                2. Upload a portrait image
                3. Choose model type and streaming options
                4. Click "Generate Video" and watch the progress
                
                ### ⚡ Streaming Mode Benefits:
                - **Faster Response**: See progress updates in real-time
                - **Better User Experience**: No more waiting in the dark
                - **Efficient Processing**: Optimized pipeline for quicker results
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter the text you want the avatar to speak...",
                        lines=5
                    )
                    
                    image_input = gr.Image(
                        label="Upload Portrait Image",
                        type="pil"
                    )
                    
                    with gr.Row():
                        use_pytorch = gr.Checkbox(
                            label="Use PyTorch Model",
                            info="Slower but more compatible",
                            value=False
                        )
                        
                        enable_streaming = gr.Checkbox(
                            label="Enable Streaming Mode",
                            info="Faster processing with progress updates",
                            value=True
                        )
                    
                    generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    status_display = gr.Textbox(
                        label="Status",
                        value="Ready to generate video",
                        interactive=False,
                        lines=2
                    )
                    
                    video_output = gr.Video(
                        label="Generated Video"
                    )
            
            # Progress information
            gr.Markdown(
                """
                ### 📊 Processing Information:
                - **Standard Mode**: Process entire video before showing result
                - **Streaming Mode**: Show progress updates and optimized processing
                - **PyTorch Model**: More compatible, works on all GPUs
                - **TensorRT Model**: Faster processing on compatible GPUs (Ampere+)
                """
            )
            
            # Example texts for reference
            gr.Markdown(
                """
                ### 💡 Example texts you can try:
                - "Hello! I am a digital avatar created with the Ditto talking head model. This is amazing AI technology!"
                - "Welcome to the future of AI-powered video generation. With streaming support, you can see results faster!"
                - "This streaming interface provides real-time progress updates while generating your talking head video."
                """
            )
            
            # Connect the generate button
            generate_btn.click(
                fn=self.process_video_streaming,
                inputs=[text_input, image_input, use_pytorch, enable_streaming],
                outputs=[video_output, status_display]
            )
            
            gr.Markdown(
                """
                ---
                ### 🔧 Technical Notes:
                - **Streaming Mode**: Uses online processing pipeline for faster results
                - **Progress Updates**: Real-time feedback during video generation
                - **Model Selection**: TensorRT is automatically selected when available
                - **Timeout**: Maximum processing time is 5 minutes
                - **GPU Requirements**: Best performance on RTX 3090 or A100 GPUs
                
                ### 📈 Performance Comparison:
                | Mode | Speed | Progress Updates | Resource Usage |
                |------|-------|------------------|----------------|
                | Standard | Slower | No | Higher memory |
                | Streaming | Faster | Yes | Optimized |
                """
            )
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )


def main():
    """Main function to run the streaming web UI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ditto Talking Head Streaming Web UI")
    parser.add_argument(
        "--runpod-endpoint",
        type=str,
        required=True,
        help="RunPod serverless endpoint URL (e.g., https://api.runpod.ai/v2/YOUR_ENDPOINT_ID)"
    )
    parser.add_argument(
        "--runpod-api-key",
        type=str,
        required=True,
        help="RunPod API key for authentication"
    )
    
    args = parser.parse_args()
    
    # Create and launch the streaming web UI
    ui = DittoStreamingWebUI(args.runpod_endpoint, args.runpod_api_key)
    ui.launch()


if __name__ == "__main__":
    # For testing, you can also set these as environment variables
    import os
    
    # Check if environment variables are set
    endpoint = os.getenv("RUNPOD_ENDPOINT")
    api_key = os.getenv("RUNPOD_API_KEY")
    
    if endpoint and api_key:
        ui = DittoStreamingWebUI(endpoint, api_key)
        ui.launch()
    else:
        # Run with command line arguments
        main()

