import os
import sys
import yaml
import json
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load StreamDiffusion configuration from YAML or JSON file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"load_config: Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_data = json.load(f)
        else:
            raise ValueError(f"load_config: Unsupported configuration file format: {config_path.suffix}")
    
    _validate_config(config_data)
    
    return config_data


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save StreamDiffusion configuration to YAML or JSON file"""
    config_path = Path(config_path)
    
    _validate_config(config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"save_config: Unsupported configuration file format: {config_path.suffix}")

def create_wrapper_from_config(config: Dict[str, Any], **overrides) -> Any:
    """Create StreamDiffusionWrapper from configuration dictionary"""
    from utils.wrapper import StreamDiffusionWrapper
    import torch
    
    final_config = {**config, **overrides}
    wrapper_params = _extract_wrapper_params(final_config)
    wrapper = StreamDiffusionWrapper(**wrapper_params)
    
    # Setup IPAdapter if configured
    if 'ipadapters' in final_config and final_config['ipadapters']:
        wrapper = _setup_ipadapter_from_config(wrapper, final_config)
    
    # Extract prepare() parameters
    prepare_params = _extract_prepare_params(final_config)
    
    if prepare_params.get('prompt'):
        wrapper.prepare(**prepare_params)
    
    return wrapper


def _extract_wrapper_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters for StreamDiffusionWrapper.__init__() from config"""
    import torch
    
    param_map = {
        'model_id_or_path': config.get('model_id', 'stabilityai/sd-turbo'),
        't_index_list': config.get('t_index_list', [0, 16, 32, 45]),
        'lora_dict': config.get('lora_dict'),
        'mode': config.get('mode', 'img2img'),
        'output_type': config.get('output_type', 'pil'),
        'lcm_lora_id': config.get('lcm_lora_id'),
        'vae_id': config.get('vae_id'),
        'device': config.get('device', 'cuda'),
        'dtype': _parse_dtype(config.get('dtype', 'float16')),
        'frame_buffer_size': config.get('frame_buffer_size', 1),
        'width': config.get('width', 512),
        'height': config.get('height', 512),
        'warmup': config.get('warmup', 10),
        'acceleration': config.get('acceleration', 'tensorrt'),
        'do_add_noise': config.get('do_add_noise', True),
        'device_ids': config.get('device_ids'),
        'use_lcm_lora': config.get('use_lcm_lora', True),
        'use_tiny_vae': config.get('use_tiny_vae', True),
        'enable_similar_image_filter': config.get('enable_similar_image_filter', False),
        'similar_image_filter_threshold': config.get('similar_image_filter_threshold', 0.98),
        'similar_image_filter_max_skip_frame': config.get('similar_image_filter_max_skip_frame', 10),
        'use_denoising_batch': config.get('use_denoising_batch', True),
        'cfg_type': config.get('cfg_type', 'self'),
        'seed': config.get('seed', 2),
        'use_safety_checker': config.get('use_safety_checker', False),
        'engine_dir': config.get('engine_dir', 'engines'),
    }
    
    if 'controlnets' in config and config['controlnets']:
        param_map['use_controlnet'] = True
        param_map['controlnet_config'] = _prepare_controlnet_configs(config)
    else:
        param_map['use_controlnet'] = config.get('use_controlnet', False)
        param_map['controlnet_config'] = config.get('controlnet_config')
    
    return {k: v for k, v in param_map.items() if v is not None}


def _extract_prepare_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters for wrapper.prepare() from config"""
    return {
        'prompt': config.get('prompt', ''),
        'negative_prompt': config.get('negative_prompt', ''),
        'num_inference_steps': config.get('num_inference_steps', 50),
        'guidance_scale': config.get('guidance_scale', 1.2),
        'delta': config.get('delta', 1.0),
    }


def _prepare_controlnet_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prepare ControlNet configurations for wrapper"""
    controlnet_configs = []
    pipeline_type = config.get('pipeline_type', 'sd1.5')
    
    for cn_config in config['controlnets']:
        controlnet_config = {
            'model_id': cn_config['model_id'],
            'preprocessor': cn_config.get('preprocessor', 'passthrough'),
            'conditioning_scale': cn_config.get('conditioning_scale', 1.0),
            'enabled': cn_config.get('enabled', True),
            'preprocessor_params': cn_config.get('preprocessor_params'),
            'pipeline_type': pipeline_type,
            'control_guidance_start': cn_config.get('control_guidance_start', 0.0),
            'control_guidance_end': cn_config.get('control_guidance_end', 1.0),
        }
        controlnet_configs.append(controlnet_config)
    
    return controlnet_configs


def _prepare_ipadapter_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prepare IPAdapter configurations for wrapper"""
    ipadapter_configs = []
    
    for ip_config in config['ipadapters']:
        ipadapter_config = {
            'ipadapter_model_path': ip_config['ipadapter_model_path'],
            'image_encoder_path': ip_config['image_encoder_path'],
            'style_image': ip_config.get('style_image'),
            'scale': ip_config.get('scale', 1.0),
            'enabled': ip_config.get('enabled', True),
        }
        ipadapter_configs.append(ipadapter_config)
    
    return ipadapter_configs


def _setup_ipadapter_from_config(wrapper, config: Dict[str, Any]):
    """Setup IPAdapter pipeline from configuration"""
    print("_setup_ipadapter_from_config: Starting IPAdapter setup...")
    print(f"_setup_ipadapter_from_config: Python path: {sys.path[:3]}...")  # Show first 3 entries
    print(f"_setup_ipadapter_from_config: Current working directory: {os.getcwd()}")
    
    try:
        print("_setup_ipadapter_from_config: Attempting to import IPAdapterPipeline...")
        
        # Add Diffusers_IPAdapter to path before importing
        import pathlib
        current_file = pathlib.Path(__file__)
        # Diffusers_IPAdapter is now located in the ipadapter directory
        diffusers_ipadapter_path = current_file.parent.parent / "ipadapter" / "Diffusers_IPAdapter"
        print(f"_setup_ipadapter_from_config: Adding Diffusers_IPAdapter to path: {diffusers_ipadapter_path}")
        print(f"_setup_ipadapter_from_config: Diffusers_IPAdapter exists: {diffusers_ipadapter_path.exists()}")
        
        if diffusers_ipadapter_path.exists():
            sys.path.insert(0, str(diffusers_ipadapter_path))
            print("_setup_ipadapter_from_config: Successfully added Diffusers_IPAdapter to Python path")
        else:
            print("_setup_ipadapter_from_config: WARNING: Diffusers_IPAdapter directory not found!")
            print(f"_setup_ipadapter_from_config: Expected location: {diffusers_ipadapter_path}")
        
        # Import here to avoid circular imports
        from ..ipadapter import IPAdapterPipeline
        print("_setup_ipadapter_from_config: Successfully imported IPAdapterPipeline")
        
        import torch
        print("_setup_ipadapter_from_config: Successfully imported torch")
        
        # Create IPAdapter pipeline
        device = config.get('device', 'cuda')
        dtype = _parse_dtype(config.get('dtype', 'float16'))
        print(f"_setup_ipadapter_from_config: Creating IPAdapterPipeline with device={device}, dtype={dtype}")
        
        ipadapter_pipeline = IPAdapterPipeline(
            stream_diffusion=wrapper.stream,
            device=device,
            dtype=dtype
        )
        print("_setup_ipadapter_from_config: Successfully created IPAdapterPipeline")
        
        # Add each configured IPAdapter
        ipadapter_configs = _prepare_ipadapter_configs(config)
        print(f"_setup_ipadapter_from_config: Found {len(ipadapter_configs)} IPAdapter configs")
        
        for i, ip_config in enumerate(ipadapter_configs):
            if ip_config.get('enabled', True):
                print(f"_setup_ipadapter_from_config: Adding IPAdapter {i}: {ip_config['ipadapter_model_path']}")
                ipadapter_pipeline.add_ipadapter(
                    ipadapter_model_path=ip_config['ipadapter_model_path'],
                    image_encoder_path=ip_config['image_encoder_path'],
                    style_image=ip_config.get('style_image'),
                    scale=ip_config.get('scale', 1.0)
                )
                print(f"_setup_ipadapter_from_config: Successfully added IPAdapter {i}")
            else:
                print(f"_setup_ipadapter_from_config: Skipping disabled IPAdapter {i}")
        
        # Replace wrapper with IPAdapter-enabled pipeline
        # Copy wrapper attributes to maintain compatibility
        ipadapter_pipeline.batch_size = getattr(wrapper, 'batch_size', 1)
        
        # Store reference to original wrapper for attribute forwarding
        ipadapter_pipeline._original_wrapper = wrapper
        print("_setup_ipadapter_from_config: IPAdapter setup completed successfully")
        
        return ipadapter_pipeline
        
    except ImportError as e:
        print(f"_setup_ipadapter_from_config: ImportError - {e}")
        print(f"_setup_ipadapter_from_config: Failed to import IPAdapter module")
        print("_setup_ipadapter_from_config: Checking if IPAdapter directory exists...")
        
        # Check if the ipadapter directory exists
        import pathlib
        current_file = pathlib.Path(__file__)
        ipadapter_path = current_file.parent.parent / "ipadapter"
        print(f"_setup_ipadapter_from_config: Looking for IPAdapter at: {ipadapter_path}")
        print(f"_setup_ipadapter_from_config: IPAdapter directory exists: {ipadapter_path.exists()}")
        
        if ipadapter_path.exists():
            print(f"_setup_ipadapter_from_config: Contents of IPAdapter directory:")
            try:
                for item in ipadapter_path.iterdir():
                    print(f"_setup_ipadapter_from_config:   - {item.name}")
            except Exception as dir_e:
                print(f"_setup_ipadapter_from_config: Error listing directory: {dir_e}")
        
        print("_setup_ipadapter_from_config: IPAdapter not available, skipping IPAdapter setup")
        return wrapper
    except Exception as e:
        print(f"_setup_ipadapter_from_config: Unexpected error - {type(e).__name__}: {e}")
        import traceback
        print("_setup_ipadapter_from_config: Full traceback:")
        traceback.print_exc()
        print("_setup_ipadapter_from_config: Falling back to wrapper without IPAdapter")
        return wrapper


def _parse_dtype(dtype_str: str) -> Any:
    """Parse dtype string to torch dtype"""
    import torch
    
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'half': torch.float16,
        'float': torch.float32,
    }
    
    if isinstance(dtype_str, str):
        return dtype_map.get(dtype_str.lower(), torch.float16)
    return dtype_str  # Assume it's already a torch dtype


def _validate_config(config: Dict[str, Any]) -> None:
    """Basic validation of configuration dictionary"""
    if not isinstance(config, dict):
        raise ValueError("_validate_config: Configuration must be a dictionary")
    
    if 'model_id' not in config:
        raise ValueError("_validate_config: Missing required field: model_id")
    
    if 'controlnets' in config:
        if not isinstance(config['controlnets'], list):
            raise ValueError("_validate_config: 'controlnets' must be a list")
        
        for i, controlnet in enumerate(config['controlnets']):
            if not isinstance(controlnet, dict):
                raise ValueError(f"_validate_config: ControlNet {i} must be a dictionary")
            
            if 'model_id' not in controlnet:
                raise ValueError(f"_validate_config: ControlNet {i} missing required 'model_id'")
    
    # Validate ipadapters if present
    if 'ipadapters' in config:
        if not isinstance(config['ipadapters'], list):
            raise ValueError("_validate_config: 'ipadapters' must be a list")
        
        for i, ipadapter in enumerate(config['ipadapters']):
            if not isinstance(ipadapter, dict):
                raise ValueError(f"_validate_config: IPAdapter {i} must be a dictionary")
            
            if 'ipadapter_model_path' not in ipadapter:
                raise ValueError(f"_validate_config: IPAdapter {i} missing required 'ipadapter_model_path'")
            
            if 'image_encoder_path' not in ipadapter:
                raise ValueError(f"_validate_config: IPAdapter {i} missing required 'image_encoder_path'")


# For backwards compatibility, provide simple functions that match expected usage patterns
def get_controlnet_config(config_dict: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    Get a specific ControlNet configuration by index
    
    Args:
        config_dict: Full configuration dictionary
        index: Index of the ControlNet to get
        
    Returns:
        ControlNet configuration dictionary
    """
    if 'controlnets' not in config_dict or index >= len(config_dict['controlnets']):
        raise IndexError(f"get_controlnet_config: ControlNet index {index} out of range")
    
    return config_dict['controlnets'][index]


def get_pipeline_type(config_dict: Dict[str, Any]) -> str:
    """
    Get pipeline type from configuration, with fallback to SD 1.5
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Pipeline type string
    """
    return config_dict.get('pipeline_type', 'sd1.5')


def get_ipadapter_config(config_dict: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    Get a specific IPAdapter configuration by index
    
    Args:
        config_dict: Full configuration dictionary
        index: Index of the IPAdapter to get
        
    Returns:
        IPAdapter configuration dictionary
    """
    if 'ipadapters' not in config_dict or index >= len(config_dict['ipadapters']):
        raise IndexError(f"get_ipadapter_config: IPAdapter index {index} out of range")
    
    return config_dict['ipadapters'][index]


def load_ipadapter_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load IPAdapter configuration from YAML or JSON file
    
    Alias for load_config() for consistency with ControlNet naming
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    return load_config(config_path)


def save_ipadapter_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save IPAdapter configuration to YAML or JSON file
    
    Alias for save_config() for consistency with ControlNet naming
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
    """
    save_config(config, config_path) 