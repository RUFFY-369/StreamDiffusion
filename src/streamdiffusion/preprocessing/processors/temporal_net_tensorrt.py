import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import Any
from .base import PipelineAwareProcessor

# Try to import TensorRT dependencies
try:
    import tensorrt as trt
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes
    from collections import OrderedDict
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Try to import torchvision for RAFT model
try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    from torchvision.utils import flow_to_image
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


class TensorRTEngine:
    """TensorRT engine wrapper for RAFT optical flow inference"""
    
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors = OrderedDict()
        self._cuda_stream = None

    def load(self):
        """Load TensorRT engine from file"""
        logger.info(f"TensorRTEngine.load: Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self):
        """Create execution context"""
        self.context = self.engine.create_execution_context()
        self._cuda_stream = torch.cuda.current_stream().cuda_stream

    def allocate_buffers(self, device="cuda"):
        """Allocate input/output buffers"""
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[name] = tensor

    def infer(self, feed_dict, stream=None):
        """Run inference with optional stream parameter"""
        if stream is None:
            stream = self._cuda_stream
            
        # Copy input data to tensors
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        # Set tensor addresses
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        
        # Execute inference
        success = self.context.execute_async_v3(stream)
        if not success:
            raise ValueError("TensorRT inference failed.")
        
        return self.tensors


class TemporalNetTensorRTPreprocessor(PipelineAwareProcessor):
    """
    TensorRT-accelerated TemporalNet preprocessor for temporal consistency using optical flow.
    
    This preprocessor uses TensorRT to accelerate RAFT optical flow computation, providing
    significant speedup over the standard PyTorch implementation.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "TemporalNet TensorRT",
            "description": "TensorRT-accelerated optical flow computation for temporal consistency in video generation.",
            "parameters": {
                "flow_strength": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.0, 2.0],
                    "step": 0.1,
                    "description": "Strength of optical flow warping (1.0 = normal, higher = more warping)"
                },
                "detect_resolution": {
                    "type": "int",
                    "default": 512,
                    "range": [256, 1024],
                    "step": 64,
                    "description": "Resolution for optical flow computation (affects quality vs speed)"
                },
                "output_format": {
                    "type": "str", 
                    "default": "concat",
                    "options": ["concat", "warped_only"],
                    "description": "Output format: 'concat' for 6-channel (current+warped), 'warped_only' for 3-channel warped frame"
                },
                "enable_tensorrt": {
                    "type": "bool",
                    "default": True,
                    "description": "Use TensorRT acceleration for optical flow computation"
                },
                "force_rebuild": {
                    "type": "bool",
                    "default": False,
                    "description": "Force rebuild TensorRT engine even if it exists"
                }
            },
            "use_cases": ["High-performance video generation", "Real-time temporal consistency", "GPU-optimized motion control"]
        }
    
    def __init__(self, 
                 pipeline_ref: Any,
                 image_resolution: int = 512,
                 flow_strength: float = 1.0,
                 detect_resolution: int = 512,
                 output_format: str = "concat",
                 enable_tensorrt: bool = True,
                 force_rebuild: bool = False,
                 **kwargs):
        """
        Initialize TensorRT TemporalNet preprocessor
        
        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (required)
            image_resolution: Output image resolution
            flow_strength: Strength of optical flow warping
            detect_resolution: Resolution for optical flow computation
            output_format: "concat" for 6-channel TemporalNetV2, "warped_only" for 3-channel
            enable_tensorrt: Use TensorRT acceleration
            force_rebuild: Force rebuild TensorRT engine
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for TemporalNet preprocessing. "
                "Install it with: pip install torchvision"
            )
        
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT and polygraphy are required for TensorRT acceleration. "
                "Install them with: pip install tensorrt polygraphy"
            )
        
        super().__init__(
            pipeline_ref=pipeline_ref,
            image_resolution=image_resolution,
            flow_strength=flow_strength,
            detect_resolution=detect_resolution,
            output_format=output_format,
            enable_tensorrt=enable_tensorrt,
            force_rebuild=force_rebuild,
            **kwargs
        )
        
        self.flow_strength = max(0.0, min(2.0, flow_strength))
        self.detect_resolution = detect_resolution
        self.enable_tensorrt = enable_tensorrt and TENSORRT_AVAILABLE
        self.force_rebuild = force_rebuild
        self._first_frame = True
        
        # Model paths
        self.models_dir = Path("models") / "temporal_net"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_path = self.models_dir / "raft_small.onnx"
        self.engine_path = self.models_dir / f"raft_small_{trt.__version__ if TENSORRT_AVAILABLE else 'notrt'}_{detect_resolution}.trt"
        
        # Model state
        self.trt_engine = None
        
        # Cached tensors for performance
        self._grid_cache = {}
        self._tensor_cache = {}
        
        # Initialize TensorRT engine
        self._ensure_model_ready()
    
    def _ensure_model_ready(self):
        """Ensure TensorRT engine is ready"""
        if not self.enable_tensorrt:
            raise RuntimeError("TemporalNetTensorRTPreprocessor requires TensorRT acceleration. Use the standard TemporalNetPreprocessor for PyTorch fallback.")
        self._setup_tensorrt()
    
    def _load_raft_for_export(self):
        """Load RAFT model temporarily for ONNX export only"""
        logger.info("_load_raft_for_export: Loading RAFT Small model for ONNX export")
        raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
        raft_model = raft_model.to(device=self.device)
        raft_model.eval()
        return raft_model
    
    def _setup_tensorrt(self):
        """Setup TensorRT engine"""
        # Export to ONNX first if needed
        if not self.onnx_path.exists() or self.force_rebuild:
            self._export_to_onnx()
        
        # Build/load TensorRT engine
        self._load_tensorrt_engine()
    
    def _export_to_onnx(self):
        """Export RAFT model to ONNX format"""
        logger.info(f"_export_to_onnx: Exporting RAFT model to ONNX: {self.onnx_path}")
        
        # Load PyTorch model temporarily for export
        raft_model = self._load_raft_for_export()
        
        # Create dummy inputs for export
        dummy_frame1 = torch.randn(1, 3, self.detect_resolution, self.detect_resolution).to(self.device)
        dummy_frame2 = torch.randn(1, 3, self.detect_resolution, self.detect_resolution).to(self.device)
        
        # Apply RAFT preprocessing if available
        weights = Raft_Small_Weights.DEFAULT
        if hasattr(weights, 'transforms') and weights.transforms is not None:
            transforms = weights.transforms()
            dummy_frame1, dummy_frame2 = transforms(dummy_frame1, dummy_frame2)
        
        dynamic_axes = {
            "frame1": {0: "batch_size"},
            "frame2": {0: "batch_size"},
            "flow": {0: "batch_size"},
        }
        
        with torch.no_grad():
            torch.onnx.export(
                raft_model,
                (dummy_frame1, dummy_frame2),
                str(self.onnx_path),
                verbose=False,
                input_names=['frame1', 'frame2'],
                output_names=['flow'],
                opset_version=17,
                export_params=True,
                dynamic_axes=dynamic_axes,
            )
        
        # Clean up the temporary model
        del raft_model
        torch.cuda.empty_cache()
        
        logger.info(f"_export_to_onnx: Successfully exported ONNX model to {self.onnx_path}")
    
    def _load_tensorrt_engine(self):
        """Load or build TensorRT engine"""
        if self.engine_path.exists() and not self.force_rebuild:
            logger.info(f"_load_tensorrt_engine: Loading existing TensorRT engine: {self.engine_path}")
            self._load_existing_engine()
        else:
            logger.info("_load_tensorrt_engine: Building new TensorRT engine")
            self._build_tensorrt_engine()
    
    def _load_existing_engine(self):
        """Load existing TensorRT engine"""
        try:
            self.trt_engine = TensorRTEngine(str(self.engine_path))
            self.trt_engine.load()
            self.trt_engine.activate()
            self.trt_engine.allocate_buffers(device=self.device)
            logger.info(f"_load_existing_engine: TensorRT engine loaded successfully from {self.engine_path}")
        except Exception as e:
            logger.error(f"_load_existing_engine: Failed to load TensorRT engine: {e}")
            self.trt_engine = None
            raise RuntimeError(f"Failed to load TensorRT engine: {e}")
    
    def _build_tensorrt_engine(self):
        """Build TensorRT engine from ONNX model"""
        if not self.onnx_path.exists():
            logger.error("TemporalNetTensorRTPreprocessor._build_tensorrt_engine: ONNX model not found")
            return
        
        logger.info("_build_tensorrt_engine: Building TensorRT engine... this may take several minutes")
        
        try:
            # Create builder and network
            builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
            
            # Parse ONNX model
            with open(self.onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("_build_tensorrt_engine: Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(f"_build_tensorrt_engine: {parser.get_error(error)}")
                    return
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 for better performance
            
            # Set optimization profile for dynamic shapes
            profile = builder.create_optimization_profile()
            min_shape = (1, 3, self.detect_resolution, self.detect_resolution)
            opt_shape = (1, 3, self.detect_resolution, self.detect_resolution)
            max_shape = (1, 3, self.detect_resolution, self.detect_resolution)
            
            profile.set_shape("frame1", min_shape, opt_shape, max_shape)
            profile.set_shape("frame2", min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            # Build engine
            engine = builder.build_serialized_network(network, config)
            
            if engine is None:
                logger.error("_build_tensorrt_engine: Failed to build TensorRT engine")
                return
            
            # Save engine
            with open(self.engine_path, 'wb') as f:
                f.write(engine)
            
            # Load the built engine
            self._load_existing_engine()
            logger.info(f"_build_tensorrt_engine: Successfully built and saved TensorRT engine: {self.engine_path}")
        
        except Exception as e:
            logger.error(f"_build_tensorrt_engine: Failed to build TensorRT engine: {e}")
            self.trt_engine = None
            raise RuntimeError(f"Failed to build TensorRT engine: {e}")
    

    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Process using TensorRT-accelerated optical flow warping
        
        Args:
            image: Current input image
            
        Returns:
            Warped previous frame for temporal guidance, or fallback for first frame
        """
        # Convert to tensor and use tensor processing path for efficiency
        tensor = self.pil_to_tensor(image)
        result_tensor = self._process_tensor_core(tensor)
        return self.tensor_to_pil(result_tensor)
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process using TensorRT-accelerated optical flow warping (GPU-optimized path)
        
        Args:
            tensor: Current input tensor
            
        Returns:
            Warped previous frame tensor for temporal guidance
        """
        
        # Check if we have a pipeline reference and previous output
        if (self.pipeline_ref is not None and 
            hasattr(self.pipeline_ref, 'prev_image_result') and 
            self.pipeline_ref.prev_image_result is not None and
            not self._first_frame):
            
            prev_output = self.pipeline_ref.prev_image_result
            
            # Convert from VAE output format [-1, 1] to [0, 1]
            prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)
            
            # Normalize input tensor
            input_tensor = tensor
            if input_tensor.max() > 1.0:
                input_tensor = input_tensor / 255.0
            
            # Ensure consistent format
            if prev_output.dim() == 4 and prev_output.shape[0] == 1:
                prev_output = prev_output[0]
            if input_tensor.dim() == 4 and input_tensor.shape[0] == 1:
                input_tensor = input_tensor[0]
            
            try:
                # Compute optical flow and warp on GPU using TensorRT
                warped_tensor = self._compute_and_warp_tensor(input_tensor, prev_output)
                
                # Check output format
                output_format = self.params.get('output_format', 'concat')
                if output_format == "concat":
                    # Concatenate current frame + warped frame for TemporalNet2 (6 channels)
                    result_tensor = self._concatenate_frames_tensor(input_tensor, warped_tensor)
                else:
                    # Return only warped frame (3 channels)
                    result_tensor = warped_tensor
                
                # Ensure correct output format
                if result_tensor.dim() == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                
                result = result_tensor.to(device=self.device, dtype=self.dtype)
            except Exception as e:
                logger.error(f"_process_tensor_core: TensorRT optical flow failed: {e}")
                output_format = self.params.get('output_format', 'concat')
                if output_format == "concat":
                    # Create 6-channel fallback by concatenating current frame with itself
                    result_tensor = self._concatenate_frames_tensor(input_tensor, input_tensor)
                    if result_tensor.dim() == 3:
                        result_tensor = result_tensor.unsqueeze(0)
                    result = result_tensor.to(device=self.device, dtype=self.dtype)
                else:
                    # Create 6-channel fallback by concatenating current frame with itself
                    result_tensor = self._concatenate_frames_tensor(input_tensor, input_tensor)
                    if result_tensor.dim() == 3:
                        result_tensor = result_tensor.unsqueeze(0)
                    result = result_tensor.to(device=self.device, dtype=self.dtype)
        else:
            # First frame or no previous output available
            self._first_frame = False
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            
            # Handle 6-channel output for first frame
            output_format = self.params.get('output_format', 'concat')
            if output_format == "concat":
                # For first frame, duplicate current frame to create 6-channel output
                if tensor.dim() == 4 and tensor.shape[0] == 1:
                    current_tensor = tensor[0]
                else:
                    current_tensor = tensor
                result_tensor = self._concatenate_frames_tensor(current_tensor, current_tensor)
                if result_tensor.dim() == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                result = result_tensor.to(device=self.device, dtype=self.dtype)
            else:
                # Create 6-channel fallback by concatenating current frame with itself
                if tensor.dim() == 4 and tensor.shape[0] == 1:
                    current_tensor = tensor[0]
                else:
                    current_tensor = tensor
                result_tensor = self._concatenate_frames_tensor(current_tensor, current_tensor)
                if result_tensor.dim() == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                result = result_tensor.to(device=self.device, dtype=self.dtype)
        
        return result
    
    def _compute_and_warp_tensor(self, current_tensor: torch.Tensor, prev_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow using TensorRT and warp previous tensor
        
        Args:
            current_tensor: Current input frame tensor (CHW format, [0,1]) on GPU
            prev_tensor: Previous pipeline output tensor (CHW format, [0,1]) on GPU
            
        Returns:
            Warped previous frame tensor on GPU
        """
        target_width, target_height = self.get_target_dimensions()
        
        # Convert to float32 for TensorRT processing
        current_tensor = current_tensor.to(device=self.device, dtype=torch.float32)
        prev_tensor = prev_tensor.to(device=self.device, dtype=torch.float32)
        
        # Resize for flow computation if needed (keep on GPU)
        if current_tensor.shape[-1] != self.detect_resolution or current_tensor.shape[-2] != self.detect_resolution:
            current_resized = F.interpolate(
                current_tensor.unsqueeze(0), 
                size=(self.detect_resolution, self.detect_resolution),
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            prev_resized = F.interpolate(
                prev_tensor.unsqueeze(0),
                size=(self.detect_resolution, self.detect_resolution), 
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            current_resized = current_tensor
            prev_resized = prev_tensor
        
        # Compute optical flow using TensorRT
        flow = self._compute_optical_flow_tensorrt(current_resized, prev_resized)
        
        # Apply flow strength scaling (GPU operation)
        flow_strength = self.params.get('flow_strength', 1.0)
        if flow_strength != 1.0:
            flow = flow * flow_strength
        
        # Warp previous frame using flow (GPU operation)
        warped_frame = self._warp_frame_tensor(prev_resized, flow)
        
        # Resize back to target resolution if needed (keep on GPU)
        if warped_frame.shape[-1] != target_width or warped_frame.shape[-2] != target_height:
            warped_frame = F.interpolate(
                warped_frame.unsqueeze(0),
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Convert to processor's dtype only at the very end
        result = warped_frame.to(dtype=self.dtype)
        
        return result
    
    def _compute_optical_flow_tensorrt(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow between two frames using TensorRT-accelerated RAFT
        
        Args:
            frame1: First frame tensor (CHW format, [0,1])
            frame2: Second frame tensor (CHW format, [0,1])
            
        Returns:
            Optical flow tensor (2HW format)
        """
        
        if self.trt_engine is None:
            raise RuntimeError("_compute_optical_flow_tensorrt: TensorRT engine not loaded")
        
        # Prepare inputs for TensorRT
        frame1_batch = frame1.unsqueeze(0)
        frame2_batch = frame2.unsqueeze(0)
        
        # Apply RAFT preprocessing if available
        weights = Raft_Small_Weights.DEFAULT
        if hasattr(weights, 'transforms') and weights.transforms is not None:
            transforms = weights.transforms()
            frame1_batch, frame2_batch = transforms(frame1_batch, frame2_batch)
        
        # Run TensorRT inference
        feed_dict = {
            'frame1': frame1_batch,
            'frame2': frame2_batch
        }
        
        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = self.trt_engine.infer(feed_dict, cuda_stream)
        flow = result['flow'][0]  # Remove batch dimension
        
        return flow
    

    
    def _warp_frame_tensor(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp frame using optical flow with cached coordinate grids
        
        Args:
            frame: Frame to warp (CHW format)
            flow: Optical flow (2HW format)
            
        Returns:
            Warped frame tensor
        """
        H, W = frame.shape[-2:]
        
        # Use cached grid if available
        grid_key = (H, W)
        if grid_key not in self._grid_cache:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=self.device, dtype=torch.float32),
                torch.arange(W, device=self.device, dtype=torch.float32),
                indexing='ij'
            )
            self._grid_cache[grid_key] = (grid_x, grid_y)
        else:
            grid_x, grid_y = self._grid_cache[grid_key]
        
        # Apply flow to coordinates
        new_x = grid_x + flow[0]
        new_y = grid_y + flow[1]
        
        # Normalize coordinates to [-1, 1] for grid_sample
        new_x = 2.0 * new_x / (W - 1) - 1.0
        new_y = 2.0 * new_y / (H - 1) - 1.0
        
        # Create sampling grid (HW2 format for grid_sample)
        grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
        
        # Warp frame
        warped_batch = F.grid_sample(
            frame.unsqueeze(0), 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        result = warped_batch.squeeze(0)
        
        return result
    
    def _concatenate_frames(self, current_image: Image.Image, warped_image: Image.Image) -> Image.Image:
        """Concatenate current frame and warped previous frame for TemporalNet2 (6-channel input)"""
        # Convert to tensors and use tensor concatenation for consistency
        current_tensor = self.pil_to_tensor(current_image).squeeze(0)
        warped_tensor = self.pil_to_tensor(warped_image).squeeze(0)
        result_tensor = self._concatenate_frames_tensor(current_tensor, warped_tensor)
        return self.tensor_to_pil(result_tensor)
    
    def _concatenate_frames_tensor(self, current_tensor: torch.Tensor, warped_tensor: torch.Tensor) -> torch.Tensor:
        """
        Concatenate current frame and warped previous frame tensors for TemporalNet2 (6-channel input)
        
        Args:
            current_tensor: Current input frame tensor (CHW format)
            warped_tensor: Warped previous frame tensor (CHW format)
            
        Returns:
            Concatenated tensor (6CHW format)
        """
        # Ensure same size
        if current_tensor.shape != warped_tensor.shape:
            target_width, target_height = self.get_target_dimensions()
            
            if current_tensor.shape[-2:] != (target_height, target_width):
                current_tensor = F.interpolate(
                    current_tensor.unsqueeze(0),
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            if warped_tensor.shape[-2:] != (target_height, target_width):
                warped_tensor = F.interpolate(
                    warped_tensor.unsqueeze(0),
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
        
        # Concatenate along channel dimension: [current_R, current_G, current_B, warped_R, warped_G, warped_B]
        concatenated = torch.cat([current_tensor, warped_tensor], dim=0)
        
        return concatenated
    
    def reset(self):
        """
        Reset the preprocessor state (useful for new sequences)
        """
        self._first_frame = True
        # Clear caches to free memory
        self._grid_cache.clear()
        self._tensor_cache.clear()
        torch.cuda.empty_cache()