# 🧭 ComfyUI Model Preset Pilot

A ComfyUI custom node module for managing model presets and configurations. This module automatically detects the model, loads/saves presets (sampler, scheduler, steps, CFG, clip-skip, seed, resolution), and generates/caches preview images per model.

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-Model_preset_Pilot.git
   ```
2. Install dependencies (if not already installed):
   ```
   pip install -r ComfyUI-Model_preset_Pilot/requirements.txt
   ```
3. Restart ComfyUI

## Features

- **Automatic Model Preset Management**: Load, save, and update presets for each model
- **Preset Parameters**: sampler, scheduler, steps, CFG, clip-skip, seed, resolution
- **Preview Generation**: Create and cache standardized preview images for each model
- **Multiple Operation Modes**: Load, Save, Update
- **Flexible Input Options**: Works with both direct conditioning or CLIP+text

## Usage

### Basic Workflow

1. Add the "🧭 Model Preset Pilot" node to your workflow
2. Connect a model (checkpoint loader) to the "model" input
3. Set the mode to "load" to retrieve saved presets for this model
4. Use the output parameters in your workflow nodes

### Modes

- **Load Mode**: Reads (or creates default) preset for the model and returns parameters + preview
- **Save Mode**: Overwrites the preset with current input values
- **Update Mode**: Updates only the fields provided, leaves others intact

### Preview Generation

To generate a preview image:
1. Connect VAE and either:
   - CLIP + positive/negative prompts, or
   - Positive/negative conditioning
2. Set "generate_preview" to True
3. Optionally set "overwrite_preview" to True to regenerate existing previews

### File Storage

- Presets are stored as JSON files in: `ComfyUI/user/model_presets/<model_id>.json`
- Preview images are stored as PNG files in: `ComfyUI/user/model_presets/previews/<model_id>.png`

## Example Workflows

### Basic Preset Loading

1. Load checkpoint → Model Preset Pilot (mode: load) → KSampler (connect all parameters)
2. Use the returned parameters to automatically configure your workflow

### Saving Custom Presets

1. Load checkpoint → Connect to Model Preset Pilot
2. Set mode to "save"
3. Input your preferred parameters
4. Run to save the preset for future use

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Project Structure

- `model_preset_pilot.py`: Main node implementation
- `__init__.py`: Node registration
- `user/model_presets/`: Directory for storing presets and previews
