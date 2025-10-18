# Data Directory

This directory contains default data and assets for the ComfyUI Model Preset Pilot.

## Structure

```
data/
├── presets/            # User-created presets (auto-generated)
│   ├── index.json      # Model registry
│   └── models/         # Individual model presets
│       └── model_name/
│           ├── metadata.json
│           ├── realistic/       # Individual preset folder
│           │   ├── preset.json
│           │   └── preview.png
│           ├── anime/          # Another preset folder
│           │   ├── preset.json
│           │   └── preview.png
│           └── portrait/       # Another preset folder
│               ├── preset.json
│               └── preview.png
├── defaults/           # Default configuration files
│   ├── preset_templates.json    # Default preset templates
│   └── storage_schema.json     # Storage schema definition
├── assets/             # Static assets (images, icons, etc.)
│   └── default_preview.png     # Default preview image
└── README.md           # This file
```

## Usage

### Default Presets
- **Location**: `data/defaults/preset_templates.json`
- **Purpose**: Contains default preset templates for different styles
- **Format**: JSON with preset configurations

### Assets
- **Location**: `data/assets/`
- **Purpose**: Static files like default preview images, icons, etc.
- **Formats**: PNG, JPG, SVG, etc.

## Adding Your Data

1. **Default Presets**: Add your preset templates to `data/defaults/`
2. **Images**: Add your default preview images to `data/assets/`
3. **Other Assets**: Add any other static files to `data/assets/`

## Loading Data

The nodes will automatically load data from this directory when needed.
