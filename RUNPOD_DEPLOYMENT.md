# Ditto Talking Head - RunPod Serverless Deployment Guide

## 🎉 Docker Image Fixed & Ready
The Docker image now includes all models and has all compatibility issues resolved

### ✅ Latest Improvements:
1. **CUDA Library Issue**: Fixed by using `cuda:11.8.0-cudnn8-devel` base image
2. **NumPy Compatibility**: Fixed by downgrading to NumPy 1.x (compatible with onnxruntime-gpu)
3. **Models Included**: Models are now baked into the Docker image (no runtime download needed)
4. **Import Error**: Fixed (AtomicCfg → StreamSDK)
5. **Model Source**: Downloads from HuggingFace during Docker build
6. **Missing Dependencies**: All added (onnxruntime-gpu, mediapipe, einops, scipy, scikit-image, scikit-learn, cython)
7. **GPU Allocation**: Removed hardcoded CUDA_VISIBLE_DEVICES for RunPod compatibility

## Prerequisites

1. **RunPod Account**: Create an account at [RunPod.io](https://runpod.io)
2. **RunPod Credits**: Add credits to your account for serverless GPU usage
3. **GitHub Account**: You'll need a GitHub account to fork the repository

## Step 1: Prepare GitHub Repository

RunPod will build the Docker image directly from your GitHub repository, eliminating the need to build and push images locally.

### Prerequisites:
1. **GitHub Repository**: Fork or clone the ditto-talkinghead repository to your GitHub account
2. **Repository Access**: Ensure your repository is public, or you have configured RunPod with access to private repositories

### Repository Setup:

The project is ready-to-deploy at: **https://github.com/mercuryyy/ditto.git**

1. **Use the Pre-configured Repository** (Recommended):
   - Repository URL: `https://github.com/mercuryyy/ditto.git` 
   - This repository is already configured with all necessary files and optimizations
   - No forking needed - you can deploy directly from this repository

2. **Or Fork for Customizations** (Optional):
   - Go to: `https://github.com/mercuryyy/ditto.git`
   - Click "Fork" to create your own copy for customizations
   - Clone your fork locally if needed:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ditto.git
   cd ditto
   ```

2. **Verify Required Files**:
   Your repository should contain:
   - `Dockerfile` (already configured for RunPod)
   - `runpod_handler.py` (RunPod serverless handler)
   - All core application files
   - `requirements_*.txt` files

3. **Optional Customizations**:
   If you need to modify any configurations, make your changes and push to GitHub:
   ```bash
   git add .
   git commit -m "Custom configuration for RunPod deployment"
   git push origin main
   ```

> **Note**: RunPod will automatically build the Docker image using the existing Dockerfile. The build includes all required models (~5GB) and takes place on RunPod's infrastructure, which is typically faster than local builds.

## Step 2: Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure the endpoint:

### Basic Configuration:
   - **Name**: `ditto-talkinghead`
   - **Source**: Select **"GitHub Repository"**
   - **GitHub URL**: `https://github.com/mercuryyy/ditto.git`
   - **Branch**: `main` (or your preferred branch)
   - **Docker Build Context**: `/` (root directory)
   - **Dockerfile Path**: `/Dockerfile`

### Hardware Configuration:
   - **GPU Type**: Select at least RTX 3090 or A100 (for TensorRT)
   - **Container Disk**: 20 GB minimum (includes pre-loaded models)
   - **Max Workers**: Start with 1-2
   - **Idle Timeout**: 5-10 seconds (models are pre-loaded, no download needed)

### Environment Variables (optional):
   ```
   LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
   ```

### Advanced Settings:
   - **Active Workers**: 0 (scale from zero)
   - **Max Workers**: 2-5 (based on expected load)
   - **Worker Timeout**: 300 seconds
   - **GPU Count**: 1

4. Click "Create Endpoint"
5. RunPod will automatically build the Docker image from your GitHub repository
6. Copy the endpoint URL and API key once the build is complete

> **Note**: RunPod builds the Docker image directly from your GitHub repository. Models are pre-loaded during the build process, so your endpoint is ready to process requests immediately once the build completes. Initial build time may take 10-15 minutes due to model downloads, but subsequent deployments with the same repository will use cached layers.

## Step 3: Test the Deployment

### Using curl:

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello, this is a test",
      "image_base64": "YOUR_BASE64_IMAGE",
      "use_pytorch": false
    }
  }'
```
### Example with request.json file:

```bash
curl -X POST https://api.runpod.ai/v2/6tz3g3hg3tq4n6/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @request.json
```

### Using Python:

```python
import requests
import base64

# Read and encode image
with open("example/image.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run",
    json={
        "input": {
            "text": "Hello, this is a test",
            "image_base64": image_base64,
            "use_pytorch": False
        }
    },
    headers={
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }
)

print(response.json())
```

## Step 4: Run the Web UI

1. Install dependencies:
```bash
pip install -r requirements_webui.txt
```

2. Set environment variables:
```bash
export RUNPOD_ENDPOINT="https://api.runpod.ai/v2/6tz3g3hg3tq4n6"
export RUNPOD_API_KEY="zzz"
```

3. Run the web UI:
```bash
python web_ui.py
```

Or with command line arguments:
```bash
python web_ui.py \
  --runpod-endpoint "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID" \
  --runpod-api-key "YOUR_API_KEY"
```

4. Open your browser to `http://localhost:7860`

## Benefits of GitHub Repository Deployment

### Advantages:
1. **Faster Deployments**: No need to build and push large Docker images locally
2. **Automatic Updates**: Push changes to GitHub and redeploy with a single click
3. **Version Control**: Easy rollbacks and version management through Git
4. **CI/CD Ready**: Can be integrated with GitHub Actions for automated deployments
5. **Collaborative**: Team members can deploy from the same repository

### Deployment Workflow:
1. Make changes to your local repository
2. Push changes to GitHub: `git push origin main`
3. In RunPod console, click "Deploy" to rebuild and redeploy
4. RunPod automatically pulls the latest code and rebuilds the container

## Cost Optimization Tips

1. **Idle Timeout**: Set a short idle timeout (5-10 seconds) to minimize costs
2. **GPU Selection**: RTX 3090 is usually more cost-effective than A100 for this workload
3. **Max Workers**: Start with 1-2 workers and scale based on demand
4. **Active Hours**: Consider scaling down during off-peak hours
5. **Build Caching**: RunPod caches Docker layers, so subsequent builds are faster

## Troubleshooting

### Issue: GitHub Repository Build Fails
**Solution**:
1. Check RunPod build logs for specific error messages
2. Ensure your Dockerfile is valid and all paths are correct
3. Verify all required files are pushed to GitHub
4. Check that your repository is public or RunPod has access permissions
5. Ensure the branch name matches what you specified in RunPod

### Issue: "Could not load library libcudnn_cnn_infer.so.8" or similar CUDA library errors
**Solution**: This has been fixed in the latest Dockerfile by using the `devel` base image instead of `runtime`. Push the updated Dockerfile to your GitHub repository and redeploy.

### Issue: NumPy compatibility error - "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.1"
**Solution**: This has been fixed in the latest Dockerfile by using `numpy<2.0` instead of `numpy==2.0.1`. Push the updated Dockerfile to your GitHub repository and redeploy.

### Issue: Build Takes Too Long
**Solution**:
1. First build from GitHub can take 15-20 minutes due to model downloads
2. Subsequent builds use cached layers and are much faster
3. Ensure your repository doesn't include unnecessary large files
4. Consider using `.dockerignore` to exclude non-essential files

### Issue: Request stuck in IN_PROGRESS state
**Solution**: 
1. Check RunPod logs for errors
2. Ensure your container has enough disk space (20GB minimum)
3. If issue persists, try setting `use_pytorch: true` in your request
4. Check that the CUDA libraries are properly loaded (LD_LIBRARY_PATH environment variable)
5. Verify NumPy version compatibility (should be < 2.0)

### Issue: TensorRT compatibility error
**Solution**: The default TensorRT models are built for Ampere+ architecture. If you get compatibility errors:
1. The handler will automatically fallback to PyTorch
2. Or explicitly use the PyTorch model by setting `use_pytorch: true`

### Issue: Out of memory error
**Solution**: 
1. Ensure you selected a GPU with at least 16GB VRAM
2. Reduce batch size in the model configuration

### Issue: Slow cold start
**Solution**:
1. Cold starts are normal after idle timeout (model initialization takes time)
2. Consider keeping 1 worker always active for faster response
3. Use TensorRT models instead of PyTorch for better performance
4. Adjust idle timeout based on your usage patterns

## Security Considerations

1. **API Key**: Never expose your RunPod API key in client-side code
2. **Input Validation**: The web UI should validate input sizes to prevent abuse
3. **Rate Limiting**: Consider implementing rate limiting in production
4. **CORS**: Configure CORS properly if hosting the web UI separately

## Advanced Configuration

### Custom TTS Integration

Replace the simple TTS in `runpod_handler.py` with more advanced options:
- Azure Cognitive Services
- Google Cloud Text-to-Speech
- Amazon Polly
- ElevenLabs

### Batch Processing

Modify the handler to support batch processing:
```python
def handler(job):
    inputs = job['input']
    if isinstance(inputs, list):
        # Process multiple requests
        results = []
        for input_item in inputs:
            result = process_single(input_item)
            results.append(result)
        return {"outputs": results}
    else:
        # Single request
        return process_single(inputs)
```

## Support

For issues specific to:
- **Ditto Model**: Check the [GitHub repository](https://github.com/antgroup/ditto-talkinghead)
- **RunPod**: Visit [RunPod Discord](https://discord.gg/runpod) or [Documentation](https://docs.runpod.io)
- **Web UI**: Check the error logs and ensure all dependencies are installed
