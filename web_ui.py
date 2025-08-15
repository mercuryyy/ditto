"""
Web UI for Ditto Talking Head using RunPod Serverless Endpoint
"""
import os
import base64
import requests
import gradio as gr
from PIL import Image
import io
import tempfile


class DittoWebUI:
    def __init__(self, runpod_endpoint, runpod_api_key):
        """
        Initialize the Web UI
        
        Args:
            runpod_endpoint: The RunPod serverless endpoint URL
            runpod_api_key: The RunPod API key for authentication
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
    
    def process_video(self, text, image, use_pytorch=False):
        """
        Process the video generation request
        
        Returns just the video path or None
        """
        if not text:
            gr.Warning("Please enter some text to synthesize.")
            return None
        
        if image is None:
            gr.Warning("Please upload an image.")
            return None
        
        try:
            # Prepare the request payload
            image_base64 = self.encode_image_to_base64(image)
            
            payload = {
                "input": {
                    "text": text,
                    "image_base64": image_base64,
                    "use_pytorch": use_pytorch
                }
            }
            
            # Send request to RunPod endpoint
            gr.Info("Sending request to RunPod...")
            response = requests.post(
                f"{self.runpod_endpoint}/run",
                json=payload,
                headers=self.headers,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code != 200:
                gr.Error(f"Failed to connect to RunPod endpoint. Status: {response.status_code}")
                return None
            
            result = response.json()
            
            # Check if the job was queued (async processing)
            if "id" in result:
                job_id = result["id"]
                gr.Info(f"Job queued with ID: {job_id}. Polling for completion...")
                # Poll for job completion
                return self.poll_job_status(job_id)
            
            # Direct response (sync processing)
            if "output" in result:
                return self.process_output(result["output"])
            
            gr.Error("Unexpected response format from RunPod")
            return None
            
        except requests.exceptions.Timeout:
            gr.Error("Request timed out. The video generation is taking too long.")
            return None
        except Exception as e:
            gr.Error(f"Error: {str(e)}")
            return None
    
    def poll_job_status(self, job_id, max_attempts=60):
        """Poll RunPod for job completion"""
        import time
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.runpod_endpoint}/status/{job_id}",
                    headers=self.headers
                )
                
                if response.status_code != 200:
                    gr.Error(f"Failed to check job status. Status: {response.status_code}")
                    return None
                
                result = response.json()
                status = result.get("status")
                
                if status == "COMPLETED":
                    gr.Info("Job completed successfully!")
                    return self.process_output(result.get("output", {}))
                elif status == "FAILED":
                    error_msg = result.get("error", "Unknown error")
                    gr.Error(f"Job failed: {error_msg}")
                    return None
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    if attempt % 3 == 0:  # Update every 15 seconds
                        gr.Info(f"Job {status}... (attempt {attempt+1}/{max_attempts})")
                    time.sleep(5)  # Wait 5 seconds before checking again
                else:
                    gr.Error(f"Unknown job status: {status}")
                    return None
                    
            except Exception as e:
                gr.Error(f"Error checking job status: {str(e)}")
                return None
        
        gr.Error("Job timed out after 5 minutes")
        return None
    
    def process_output(self, output):
        """Process the output from RunPod"""
        if "error" in output:
            gr.Error(f"Error from RunPod: {output['error']}")
            return None
        
        if "video_base64" not in output:
            gr.Error("No video data in response")
            return None
        
        try:
            # Decode the video from base64
            video_data = base64.b64decode(output["video_base64"])
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_file.write(video_data)
                gr.Info("Video generated successfully!")
                return tmp_file.name
                
        except Exception as e:
            gr.Error(f"Error processing video: {str(e)}")
            return None
    
    def launch(self):
        """Launch the Gradio interface"""
        with gr.Blocks(title="Ditto Talking Head Web UI", theme=gr.themes.Soft()) as interface:
            gr.Markdown(
                """
                # 🎭 Ditto Talking Head Web UI
                
                Generate lip-synced talking head videos from text and images using the Ditto model deployed on RunPod.
                
                ## How to use:
                1. Enter the text you want to synthesize
                2. Upload a portrait image
                3. Choose whether to use PyTorch or TensorRT model
                4. Click "Generate Video" and wait for the result
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
                    
                    use_pytorch = gr.Checkbox(
                        label="Use PyTorch Model (slower but more compatible)",
                        value=False
                    )
                    
                    generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    video_output = gr.Video(
                        label="Generated Video"
                    )
            
            # Example texts for reference
            gr.Markdown(
                """
                ### Example texts you can try:
                - "Hello! I am a digital avatar created with the Ditto talking head model. This is an amazing AI technology!"
                - "Welcome to the future of AI-powered video generation. With Ditto, you can create realistic talking head videos from just a single image!"
                """
            )
            
            # Connect the generate button - single output only
            generate_btn.click(
                fn=self.process_video,
                inputs=[text_input, image_input, use_pytorch],
                outputs=video_output
            )
            
            gr.Markdown(
                """
                ---
                ### Notes:
                - The first request might take longer as the model needs to warm up
                - Video generation typically takes 30-60 seconds depending on text length
                - For best results, use a clear portrait image with the face clearly visible
                - The TensorRT model is faster but requires compatible GPU architecture
                - Status updates will appear as notifications at the top of the screen
                """
            )
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )


def main():
    """Main function to run the web UI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ditto Talking Head Web UI")
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
    
    # Create and launch the web UI
    ui = DittoWebUI(args.runpod_endpoint, args.runpod_api_key)
    ui.launch()


if __name__ == "__main__":
    # For testing, you can also set these as environment variables
    import os
    
    # Check if environment variables are set
    endpoint = os.getenv("RUNPOD_ENDPOINT")
    api_key = os.getenv("RUNPOD_API_KEY")
    
    if endpoint and api_key:
        ui = DittoWebUI(endpoint, api_key)
        ui.launch()
    else:
        # Run with command line arguments
        main()
