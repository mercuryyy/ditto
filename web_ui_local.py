"""
Local Web UI for Ditto Talking Head (for testing without RunPod)
This version runs the model locally instead of using RunPod
"""
import os
import pickle
import tempfile
import gradio as gr
from PIL import Image
from typing import Optional
import subprocess
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from stream_pipeline_offline import process_stream_offline_main
from gtts import gTTS


class DittoLocalWebUI:
    def __init__(self, data_root: str = None, cfg_pkl: str = None):
        """
        Initialize the Local Web UI
        
        Args:
            data_root: Path to the model directory
            cfg_pkl: Path to the config pickle file
        """
        # Default paths
        if data_root is None:
            # Check which model type is available
            if os.path.exists("./checkpoints/ditto_pytorch"):
                self.data_root = "./checkpoints/ditto_pytorch"
                self.cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
                self.model_type = "PyTorch"
            elif os.path.exists("./checkpoints/ditto_trt_custom"):
                self.data_root = "./checkpoints/ditto_trt_custom"
                self.cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
                self.model_type = "TensorRT (Custom)"
            else:
                self.data_root = "./checkpoints/ditto_trt_Ampere_Plus"
                self.cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
                self.model_type = "TensorRT (Ampere+)"
        else:
            self.data_root = data_root
            self.cfg_pkl = cfg_pkl if cfg_pkl else "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
            self.model_type = "Custom"
        
        # Load configuration
        self.cfg = None
        if os.path.exists(self.cfg_pkl):
            with open(self.cfg_pkl, 'rb') as f:
                self.cfg = pickle.load(f)
        else:
            print(f"Warning: Config file not found at {self.cfg_pkl}")
    
    def text_to_speech(self, text: str, output_path: str) -> str:
        """Convert text to speech using gTTS"""
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(output_path)
            return output_path
        except Exception as e:
            print(f"TTS Error: {e}")
            # Create a silent audio file as fallback
            subprocess.run([
                'ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono', 
                '-t', '5', '-acodec', 'pcm_s16le', output_path
            ], capture_output=True)
            return output_path
    
    def process_video(self, text: str, image: Optional[Image.Image]) -> str:
        """
        Process the video generation request locally
        
        Args:
            text: Text to synthesize
            image: Input image
        
        Returns:
            Path to the generated video file or error message
        """
        if not text:
            return None
        
        if image is None:
            return None
        
        if self.cfg is None:
            return None
        
        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save the input image
                image_path = os.path.join(temp_dir, 'input_image.png')
                image.save(image_path)
                
                # Convert text to speech
                audio_path = os.path.join(temp_dir, 'input_audio.wav')
                self.text_to_speech(text, audio_path)
                
                # Output video path
                output_path = os.path.join(temp_dir, 'output.mp4')
                
                # Process the video
                process_stream_offline_main(
                    cfg=self.cfg,
                    data_root=self.data_root,
                    audio_path=audio_path,
                    source_path=image_path,
                    output_path=output_path
                )
                
                # Copy to a persistent location
                import shutil
                persistent_path = tempfile.mktemp(suffix='.mp4')
                shutil.copy(output_path, persistent_path)
                
                return persistent_path
                
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
    
    def launch(self):
        """Launch the Gradio interface"""
        with gr.Blocks(title="Ditto Talking Head Local Web UI") as interface:
            gr.Markdown(
                f"""
                # 🎭 Ditto Talking Head - Local Web UI
                
                Generate lip-synced talking head videos from text and images using the Ditto model.
                
                **Model Type**: {self.model_type}  
                **Model Path**: {self.data_root}
                
                ## How to use:
                1. Enter the text you want to synthesize
                2. Upload a portrait image
                3. Click "Generate Video" and wait for the result
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter the text you want the avatar to speak...",
                        lines=5,
                        value="Hello! I am a digital avatar created with the Ditto talking head model."
                    )
                    
                    image_input = gr.Image(
                        label="Upload Portrait Image",
                        type="pil",
                        height=300
                    )
                    
                    # Load example image if available
                    if os.path.exists("./example/image.png"):
                        gr.Markdown("💡 **Tip**: You can use the example image from `./example/image.png`")
                    
                    generate_btn = gr.Button("🎬 Generate Video", variant="primary")
                
                with gr.Column(scale=1):
                    video_output = gr.Video(
                        label="Generated Video",
                        height=400
                    )
                    
                    status_text = gr.Markdown("")
            
            # Example inputs
            example_texts = [
                "Hello! I am a digital avatar created with the Ditto talking head model. This is an amazing AI technology!",
                "Welcome to the future of AI-powered video generation. With Ditto, you can create realistic talking head videos from just a single image!",
                "The weather today is absolutely beautiful. I hope you're having a wonderful day!",
            ]
            
            gr.Examples(
                examples=[[text] for text in example_texts],
                inputs=[text_input],
                label="Example Texts"
            )
            
            def process_with_status(text, image):
                if not text:
                    gr.Warning("Please enter some text to synthesize.")
                    return None
                if image is None:
                    gr.Warning("Please upload an image.")
                    return None
                
                gr.Info("Processing... This may take 30-60 seconds.")
                result = self.process_video(text, image)
                
                if result:
                    gr.Info("Video generated successfully!")
                else:
                    gr.Error("Failed to generate video. Check the console for errors.")
                
                return result
            
            # Connect the generate button
            generate_btn.click(
                fn=process_with_status,
                inputs=[text_input, image_input],
                outputs=video_output
            )
            
            gr.Markdown(
                """
                ---
                ### Notes:
                - Video generation typically takes 30-60 seconds depending on text length
                - For best results, use a clear portrait image with the face clearly visible
                - The first run might be slower as models are loaded into memory
                - Make sure you have downloaded the model checkpoints before running
                
                ### Model Download:
                If you haven't downloaded the models yet, run:
                ```bash
                git lfs install
                git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
                ```
                """
            )
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )


def main():
    """Main function to run the local web UI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ditto Talking Head Local Web UI")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the model directory (e.g., ./checkpoints/ditto_pytorch)"
    )
    parser.add_argument(
        "--cfg-pkl",
        type=str,
        default=None,
        help="Path to the config pickle file"
    )
    
    args = parser.parse_args()
    
    # Create and launch the web UI
    ui = DittoLocalWebUI(data_root=args.data_root, cfg_pkl=args.cfg_pkl)
    ui.launch()


if __name__ == "__main__":
    main()
