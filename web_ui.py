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
from typing import Optional


class DittoWebUI:
    def __init__(self, runpod_endpoint: str, runpod_api_key: str):
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
    
    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def process_video(self, text: str, image: Optional[Image.Image], use_pytorch: bool = False) -> str:
        """
        Process the video generation request
        
        Args:
            text: Text to synthesize
            image: Input image
            use_pytorch: Whether to use PyTorch model (vs TensorRT)
        
        Returns:
            Path to the generated video file
        """
        if not text:
            return "Please enter some text to synthesize."
        
        if image is None:
            return "Please upload an image."
        
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
            response = requests.post(
                f"{self.runpod_endpoint}/run",
                json=payload,
                headers=self.headers,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code != 200:
                return f"Error: Failed to connect to RunPod endpoint. Status: {response.status_code}"
            
            result = response.json()
            
            # Check if the job was queued (async processing)
            if "id" in result:
                job_id = result["id"]
                # Poll for job completion
                return self.poll_job_status(job_id)
            
            # Direct response (sync processing)
            if "output" in result:
                return self.process_output(result["output"])
            
            return "Error: Unexpected response format from RunPod"
            
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The video generation is taking too long."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def poll_job_status(self, job_id: str, max_attempts: int = 60) -> str:
        """Poll RunPod for job completion"""
        import time
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.runpod_endpoint}/status/{job_id}",
                    headers=self.headers
                )
                
                if response.status_code != 200:
                    return f"Error: Failed to check job status. Status: {response.status_code}"
                
                result = response.json()
                status = result.get("status")
                
                if status == "COMPLETED":
                    return self.process_output(result.get("output", {}))
                elif status == "FAILED":
                    error_msg = result.get("error", "Unknown error")
                    return f"Error: Job failed - {error_msg}"
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    time.sleep(5)  # Wait 5 seconds before checking again
                else:
                    return f"Error: Unknown job status - {status}"
                    
            except Exception as e:
                return f"Error checking job status: {str(e)}"
        
        return "Error: Job timed out after 5 minutes"
    
    def process_output(self, output: dict) -> str:
        """Process the output from RunPod"""
        if "error" in output:
            return f"Error from RunPod: {output['error']}"
        
        if "video_base64" not in output:
            return "Error: No video data in response"
        
        try:
            # Decode the video from base64
            video_data = base64.b64decode(output["video_base64"])
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_file.write(video_data)
                return tmp_file.name
                
        except Exception as e:
            return f"Error processing video: {str(e)}"
    
    def launch(self):
        """Launch the Gradio interface"""
        with gr.Blocks(title="Ditto Talking Head Web UI") as interface:
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
                        type="pil",
                        height=300
                    )
                    
                    use_pytorch = gr.Checkbox(
                        label="Use PyTorch Model (slower but more compatible)",
                        value=False
                    )
                    
                    generate_btn = gr.Button("🎬 Generate Video", variant="primary")
                
                with gr.Column(scale=1):
                    video_output = gr.Video(
                        label="Generated Video",
                        height=400
                    )
                    
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        visible=False
                    )
            
            # Example inputs
            gr.Examples(
                examples=[
                    ["Hello! I am a digital avatar created with the Ditto talking head model. This is an amazing AI technology!", None, False],
                    ["Welcome to the future of AI-powered video generation. With Ditto, you can create realistic talking head videos from just a single image!", None, False],
                ],
                inputs=[text_input, image_input, use_pytorch],
                label="Example Texts"
            )
            
            # Connect the generate button
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
