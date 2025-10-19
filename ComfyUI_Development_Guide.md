# ComfyUI Custom Node Development Guide

## Overview

Custom nodes allow you to implement new features and share them with the wider community.
A custom node is like any Comfy node: it takes input, does something to it, and produces an output. While some custom nodes perform highly complex tasks, many just do one thing.

## Client-Server Model

Comfy runs in a client-server model:
- **Server** (Python): Handles all the real work - data-processing, models, image diffusion etc.
- **Client** (Javascript): Handles the user interface
- **API Mode**: Workflow sent to server by non-Comfy client (command line script, other UI)

## Custom Node Categories

### 1. Server Side Only
- **Majority of Custom Nodes** run purely on the server side
- Define a Python class that specifies input/output types
- Provides a function to process inputs and produce output

### 2. Client Side Only
- Provide modification to the client UI
- Do not add core functionality
- May not even add new nodes to the system

### 3. Independent Client and Server
- Provide additional server features
- Additional (related) UI features (new widget for new data type)
- Communication handled by Comfy data flow control

### 4. Connected Client and Server
- Small number of cases
- UI features and server need to interact directly

## Node Properties - Complete Reference

### Simple Example: Invert Image Node

```python
class InvertImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "image_in" : ("IMAGE", {}) },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    CATEGORY = "examples"
    FUNCTION = "invert"

    def invert(self, image_in):
        image_out = 1 - image_in
        return (image_out,)
```

### Main Properties

#### INPUT_TYPES
- **Required**: Must contain `required` key
- **Optional**: May include `optional` and/or `hidden` keys
- **Optional vs Required**: Optional inputs can be left unconnected
- **Format**: `("TYPE", {"param": "value"})` - tuple with type and parameters
- **Class Method**: Must be `@classmethod` for runtime option computation

#### RETURN_TYPES
- **Tuple of strings**: Defines output data types
- **Empty tuple**: `RETURN_TYPES = ()` for no outputs
- **Single output**: `RETURN_TYPES = ("IMAGE",)` - trailing comma required
- **Multiple outputs**: `RETURN_TYPES = ("IMAGE", "STRING", "INT")`

#### RETURN_NAMES
- **Optional**: Human-readable output labels
- **Default**: Uses RETURN_TYPES in lowercase if omitted
- **Must match**: Number of RETURN_TYPES

#### CATEGORY
- **Menu location**: Where node appears in Add Node menu
- **Submenus**: Use path format `"examples/trivial"`
- **Custom categories**: `"🤖 My Category"` for emoji support

#### FUNCTION
- **Method name**: Python function called on execution
- **Named arguments**: Receives all required and connected optional inputs
- **Default values**: Provide defaults for optional inputs
- **Return tuple**: Must match RETURN_TYPES count
- **Single output**: `return (result,)` - trailing comma required

### Execution Control

#### OUTPUT_NODE
```python
OUTPUT_NODE = True  # Node is considered an output
```
- **Default**: `False` (node is not an output)
- **Purpose**: Marks nodes that produce final results
- **Caching**: Output nodes are always executed

#### IS_CHANGED
```python
@classmethod
def IS_CHANGED(cls, input1, input2):
    # Return any object for comparison
    return hash(str(input1) + str(input2))
```
- **Purpose**: Control when node needs re-execution
- **Default**: Node changes if any input/widget changes
- **Return type**: Any Python object (not just bool!)
- **Comparison**: `is_changed != is_changed_old`
- **Always change**: `return float("NaN")` (avoid if possible)
- **Example**: File hash for LoadImage node

#### VALIDATE_INPUTS
```python
@classmethod
def VALIDATE_INPUTS(cls, input1, input2):
    if input1 < 0:
        return "input1 must be positive"
    return True
```
- **Purpose**: Validate inputs before execution
- **Return**: `True` if valid, error message (str) if invalid
- **Constants only**: Only receives constant inputs, not node outputs
- **Skip validation**: Use `**kwargs` to receive all inputs

### Advanced Validation

#### Validating Constants
```python
class CustomNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "foo" : ("INT", {"min": 0, "max": 10}) },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, foo):
        # Custom validation logic
        return True
```

#### Validating Types
```python
class AddNumbers:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1" : ("INT,FLOAT", {"min": 0, "max": 1000}),
                "input2" : ("INT,FLOAT", {"min": 0, "max": 1000})
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        if input_types["input1"] not in ("INT", "FLOAT"):
            return "input1 must be an INT or FLOAT type"
        if input_types["input2"] not in ("INT", "FLOAT"):
            return "input2 must be an INT or FLOAT type"
        return True
```

### Other Attributes

#### INPUT_IS_LIST, OUTPUT_IS_LIST
- **Purpose**: Control sequential processing of data
- **Advanced feature**: For batch processing nodes

## Hidden and Flexible Inputs

### Hidden Inputs

Hidden inputs allow custom nodes to request server-side information without creating client-side widgets.

#### Basic Structure
```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {...},
        "optional": {...},
        "hidden": {
            "unique_id": "UNIQUE_ID",
            "prompt": "PROMPT", 
            "extra_pnginfo": "EXTRA_PNGINFO",
        }
    }
```

#### UNIQUE_ID
```python
def process(self, unique_id, ...):
    # unique_id is the node's unique identifier
    print(f"Node ID: {unique_id}")
    return (result,)
```
- **Purpose**: Node's unique identifier
- **Use case**: Client-server communication
- **Type**: String

#### PROMPT
```python
def process(self, prompt, ...):
    # prompt contains the complete workflow prompt
    print(f"Full prompt: {prompt}")
    return (result,)
```
- **Purpose**: Complete workflow prompt
- **Use case**: Access to entire workflow context
- **Type**: Dictionary with workflow information

#### EXTRA_PNGINFO
```python
def process(self, extra_pnginfo, ...):
    # extra_pnginfo will be saved in PNG metadata
    extra_pnginfo["custom_data"] = "my_value"
    return (result,)
```
- **Purpose**: PNG metadata dictionary
- **Use case**: Storing custom data in saved images
- **Note**: Disabled if ComfyUI started with `--disable-metadata`

#### DYNPROMPT (Advanced)
```python
def process(self, dynprompt, ...):
    # dynprompt may mutate during execution
    # Only for advanced cases like loops
    return (result,)
```
- **Purpose**: Dynamic prompt that can mutate
- **Use case**: Node expansion and loops
- **Advanced**: Only for complex workflows

### Flexible Inputs

#### Custom Datatypes
```python
# Define custom datatype
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "my_cheese": ("CHEESE", {"forceInput": True})
        }
    }

def process(self, my_cheese, ...):
    # my_cheese can be any Python object
    print(f"Cheese type: {type(my_cheese)}")
    return (result,)
```

**Key Points:**
- **Unique name**: Use uppercase string (e.g., "CHEESE")
- **Any object**: Can be any Python object
- **forceInput**: Required to prevent widget creation
- **Type safety**: Client enforces type connections

#### Wildcard Inputs
```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "anything": ("*", {})
        }
    }

@classmethod
def VALIDATE_INPUTS(cls, input_types):
    # Skip backend validation for wildcard inputs
    return True

def process(self, anything, ...):
    # anything can be any type
    print(f"Received: {type(anything)}")
    return (result,)
```

**Key Points:**
- **Universal input**: Accepts any data type
- **Client-side**: Frontend allows any connection
- **Backend validation**: Use VALIDATE_INPUTS to skip
- **Node responsibility**: Handle any data type appropriately

#### Dynamically Created Inputs
```python
class ContainsAnyDict(dict):
    def __contains__(self, key):
        return True

@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {},
        "optional": ContainsAnyDict()
    }

def process(self, **kwargs):
    # Dynamically created inputs are in kwargs
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    return (result,)
```

**Key Points:**
- **Dynamic inputs**: Client-side created inputs
- **Arbitrary names**: Any input name allowed
- **kwargs capture**: All dynamic inputs in **kwargs
- **Pythonic trick**: Credit to rgthree

### Advanced Patterns

#### Conditional Inputs
```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "mode": (["simple", "advanced"], {}),
            "input_data": ("*", {"forceInput": True})
        }
    }

def process(self, mode, input_data, **kwargs):
    if mode == "simple":
        # Handle simple mode
        return (input_data,)
    else:
        # Handle advanced mode with dynamic inputs
        return (kwargs.get("advanced_input", input_data),)
```

#### Metadata Handling
```python
def process(self, image, extra_pnginfo, unique_id, **kwargs):
    # Store custom metadata
    if "custom_metadata" not in extra_pnginfo:
        extra_pnginfo["custom_metadata"] = {}
    
    extra_pnginfo["custom_metadata"][unique_id] = {
        "processed_at": time.time(),
        "node_type": self.__class__.__name__
    }
    
    return (image,)
```

#### Communication Between Nodes
```python
# Node A: Sender
def process(self, data, extra_pnginfo, ...):
    # Store data for other nodes
    extra_pnginfo["shared_data"] = {
        "processed_data": data,
        "timestamp": time.time()
    }
    return (data,)

# Node B: Receiver
def process(self, image, extra_pnginfo, ...):
    # Retrieve data from other nodes
    shared_data = extra_pnginfo.get("shared_data", {})
    if shared_data:
        processed_data = shared_data.get("processed_data")
        # Use the shared data
    return (image,)
```

### Best Practices

#### Input Validation
```python
@classmethod
def VALIDATE_INPUTS(cls, input_types):
    # Validate wildcard inputs
    if "anything" in input_types:
        if input_types["anything"] not in ["IMAGE", "LATENT"]:
            return "anything must be IMAGE or LATENT"
    return True
```

#### Error Handling
```python
def process(self, **kwargs):
    try:
        # Handle dynamic inputs safely
        dynamic_input = kwargs.get("dynamic_input")
        if dynamic_input is not None:
            # Process dynamic input
            pass
    except Exception as e:
        print(f"Error processing dynamic input: {e}")
        return (None,)
```

#### Documentation
```python
class MyNode:
    """
    My Custom Node with flexible inputs
    
    Hidden Inputs:
        unique_id: Node's unique identifier
        prompt: Complete workflow prompt
        extra_pnginfo: PNG metadata dictionary
    
    Dynamic Inputs:
        Any input created dynamically on the client
    """
```

### Common Use Cases

#### Workflow Metadata
```python
def process(self, image, extra_pnginfo, unique_id, ...):
    # Add workflow information to metadata
    extra_pnginfo["workflow_info"] = {
        "node_id": unique_id,
        "node_class": self.__class__.__name__,
        "timestamp": time.time()
    }
    return (image,)
```

#### Inter-Node Communication
```python
def process(self, data, extra_pnginfo, ...):
    # Store data for downstream nodes
    extra_pnginfo["upstream_data"] = data
    return (data,)
```

#### Dynamic Processing
```python
def process(self, **kwargs):
    # Process any number of dynamic inputs
    results = []
    for key, value in kwargs.items():
        if key.startswith("input_"):
            processed = self.process_input(value)
            results.append(processed)
    return (results,)
```

## Lazy Evaluation

### Purpose
Lazy evaluation allows nodes to skip evaluating inputs that aren't needed, improving performance by avoiding unnecessary processing.

### When to Use Lazy Evaluation
- **Model merging**: Skip loading models when ratio is 0.0 or 1.0
- **Image interpolation**: Skip images when mask is entirely 0.0 or 1.0
- **Switch nodes**: Skip inputs based on condition
- **Conditional processing**: Skip expensive operations when not needed

### Creating Lazy Inputs

#### Step 1: Mark Inputs as Lazy
```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "image1": ("IMAGE", {"lazy": True}),
            "image2": ("IMAGE", {"lazy": True}),
            "mask": ("MASK", {}),  # Always evaluated
        }
    }
```

#### Step 2: Define check_lazy_status
```python
def check_lazy_status(self, mask, image1, image2):
    # mask is always available, image1/image2 may be None
    mask_min = mask.min()
    mask_max = mask.max()
    needed = []
    
    # Need image1 if mask is not entirely 1.0
    if image1 is None and (mask_min != 1.0 or mask_max != 1.0):
        needed.append("image1")
    
    # Need image2 if mask is not entirely 0.0
    if image2 is None and (mask_min != 0.0 or mask_max != 0.0):
        needed.append("image2")
    
    return needed
```

### Complete Example: LazyMixImages

```python
class LazyMixImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {"lazy": True}),
                "image2": ("IMAGE", {"lazy": True}),
                "mask": ("MASK", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mix"
    CATEGORY = "Examples"

    def check_lazy_status(self, mask, image1, image2):
        mask_min = mask.min()
        mask_max = mask.max()
        needed = []
        
        if image1 is None and (mask_min != 1.0 or mask_max != 1.0):
            needed.append("image1")
        if image2 is None and (mask_min != 0.0 or mask_max != 0.0):
            needed.append("image2")
        
        return needed

    def mix(self, mask, image1, image2):
        mask_min = mask.min()
        mask_max = mask.max()
        
        # If mask is entirely 0.0, return image1
        if mask_min == 0.0 and mask_max == 0.0:
            return (image1,)
        
        # If mask is entirely 1.0, return image2
        elif mask_min == 1.0 and mask_max == 1.0:
            return (image2,)
        
        # Otherwise, interpolate
        result = image1 * (1.0 - mask) + image2 * mask
        return (result,)
```

### Key Points

#### check_lazy_status Method
- **Not a class method**: Uses actual input values
- **May be called multiple times**: As more inputs become available
- **Returns list**: Names of lazy inputs still needed
- **Empty list**: When all required inputs are available

#### Lazy Input Behavior
- **None values**: Unavailable lazy inputs are None
- **Available inputs**: Non-lazy inputs always have values
- **Multiple calls**: Function may be called repeatedly
- **Performance**: Minimal cost, should be used when possible

### Advanced Patterns

#### Conditional Model Loading
```python
class LazyModelMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL", {"lazy": True}),
                "model2": ("MODEL", {"lazy": True}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    def check_lazy_status(self, ratio, model1, model2):
        needed = []
        
        # Only need model1 if ratio < 1.0
        if model1 is None and ratio < 1.0:
            needed.append("model1")
        
        # Only need model2 if ratio > 0.0
        if model2 is None and ratio > 0.0:
            needed.append("model2")
        
        return needed

    def merge(self, ratio, model1, model2):
        if ratio == 0.0:
            return (model1,)
        elif ratio == 1.0:
            return (model2,)
        else:
            # Perform actual merging
            return (merged_model,)
```

#### Switch Node Pattern
```python
class LazySwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {}),
                "input_a": ("*", {"lazy": True}),
                "input_b": ("*", {"lazy": True}),
            }
        }

    def check_lazy_status(self, condition, input_a, input_b):
        needed = []
        
        if condition and input_a is None:
            needed.append("input_a")
        elif not condition and input_b is None:
            needed.append("input_b")
        
        return needed

    def switch(self, condition, input_a, input_b):
        return (input_a if condition else input_b,)
```

## Execution Blocking

### Purpose
Execution blocking allows nodes to prevent downstream execution when certain conditions aren't met.

### ExecutionBlocker Usage

#### Silent Blocking
```python
from comfy_execution.graph import ExecutionBlocker

def silent_passthrough(self, passthrough, blocked):
    if blocked:
        return (ExecutionBlocker(None),)
    else:
        return (passthrough,)
```

#### Error Message Blocking
```python
def load_checkpoint(self, ckpt_name):
    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    model, clip, vae = load_checkpoint(ckpt_path)
    
    if vae is None:
        # Provide meaningful error message
        vae = ExecutionBlocker(f"No VAE contained in the loaded model {ckpt_name}")
    
    return (model, clip, vae)
```

### ExecutionBlocker Properties
- **Propagates forward**: Cannot be stopped from propagating
- **Silent blocking**: Use `ExecutionBlocker(None)`
- **Error messages**: Use `ExecutionBlocker("message")`
- **No backward compatibility**: Use lazy evaluation instead

### Best Practices

#### When to Use Lazy Evaluation
- **Performance critical**: Expensive operations that may be skipped
- **Conditional logic**: Inputs only needed under certain conditions
- **Resource intensive**: Model loading, large image processing

#### When to Use Execution Blocking
- **Error conditions**: Invalid configurations that should fail
- **Missing dependencies**: Required components not available
- **Invalid states**: Workflow configurations that won't work

#### Implementation Tips
```python
def check_lazy_status(self, condition, expensive_input, ...):
    # Always check if input is None first
    if expensive_input is None:
        # Check if input is actually needed
        if condition_requires_input(condition):
            return ["expensive_input"]
    
    return []  # No more inputs needed
```

### Common Patterns

#### Model Loading with Fallback
```python
def load_model_with_fallback(self, model_name, fallback_model):
    try:
        model = load_model(model_name)
        return (model,)
    except Exception as e:
        if fallback_model is not None:
            return (fallback_model,)
        else:
            return (ExecutionBlocker(f"Failed to load {model_name}: {e}"),)
```

#### Conditional Processing
```python
def conditional_process(self, input_data, condition, expensive_input):
    if condition:
        # Process with expensive input
        return (process_with_expensive(input_data, expensive_input),)
    else:
        # Simple processing
        return (simple_process(input_data),)
```

## Node Expansion

### Purpose
Node expansion allows nodes to return a subgraph of nodes that replaces the original node in the execution graph. This enables advanced features like loops and complex workflow generation.

### When to Use Node Expansion
- **Loops**: Implementing iterative processing
- **Complex workflows**: Generating subgraphs dynamically
- **Caching optimization**: Separate caching for subgraph components
- **Workflow generation**: Creating workflows programmatically

### Basic Node Expansion

#### Simple Example: Load and Merge Checkpoints
```python
from comfy_execution.graph_utils import GraphBuilder

def load_and_merge_checkpoints(self, checkpoint_path1, checkpoint_path2, ratio):
    graph = GraphBuilder()
    
    # Create checkpoint loader nodes
    checkpoint_node1 = graph.node("CheckpointLoaderSimple", checkpoint_path=checkpoint_path1)
    checkpoint_node2 = graph.node("CheckpointLoaderSimple", checkpoint_path=checkpoint_path2)
    
    # Create merge nodes
    merge_model_node = graph.node("ModelMergeSimple", 
                                 model1=checkpoint_node1.out(0), 
                                 model2=checkpoint_node2.out(0), 
                                 ratio=ratio)
    merge_clip_node = graph.node("ClipMergeSimple", 
                                clip1=checkpoint_node1.out(1), 
                                clip2=checkpoint_node2.out(1), 
                                ratio=ratio)
    
    return {
        # Return outputs: (MODEL, CLIP, VAE)
        "result": (merge_model_node.out(0), merge_clip_node.out(0), checkpoint_node1.out(2)),
        "expand": graph.finalize(),
    }
```

### Requirements

#### Return Dictionary Format
```python
def expand_node(self, ...):
    return {
        "result": (output1, output2, output3),  # Tuple of outputs
        "expand": graph.finalize(),            # Finalized subgraph
    }
```

#### Key Requirements
- **result**: Tuple of node outputs (can mix finalized values and node outputs)
- **expand**: Finalized graph for expansion
- **Unique IDs**: Node IDs must be unique across entire graph
- **Deterministic IDs**: IDs must be consistent between executions

### GraphBuilder Usage

#### Basic Graph Creation
```python
from comfy_execution.graph_utils import GraphBuilder

def create_subgraph(self, input_data):
    graph = GraphBuilder()
    
    # Create nodes
    node1 = graph.node("NodeType1", input=input_data)
    node2 = graph.node("NodeType2", input=node1.out(0))
    node3 = graph.node("NodeType3", input=node2.out(0))
    
    # Return expanded graph
    return {
        "result": (node3.out(0),),
        "expand": graph.finalize(),
    }
```

#### Advanced Graph Creation
```python
def create_complex_subgraph(self, model, image, steps):
    graph = GraphBuilder()
    
    # Create multiple processing chains
    chain1 = graph.node("ProcessA", input=image)
    chain2 = graph.node("ProcessB", input=image)
    
    # Merge chains
    merge_node = graph.node("MergeNodes", 
                           input1=chain1.out(0), 
                           input2=chain2.out(0))
    
    # Final processing
    final_node = graph.node("FinalProcess", 
                           input=merge_node.out(0), 
                           model=model)
    
    return {
        "result": (final_node.out(0),),
        "expand": graph.finalize(),
    }
```

### Manual Graph Creation (Without GraphBuilder)

#### Basic Manual Format
```python
def manual_expansion(self, input_data):
    # Generate unique prefix
    prefix = GraphBuilder.alloc_prefix()
    
    # Create graph manually
    graph = {
        f"{prefix}_node1": {
            "class_type": "NodeType1",
            "inputs": {"input": input_data}
        },
        f"{prefix}_node2": {
            "class_type": "NodeType2", 
            "inputs": {"input": [f"{prefix}_node1", 0]}
        }
    }
    
    return {
        "result": ([f"{prefix}_node2", 0],),
        "expand": graph,
    }
```

#### Using GraphBuilder for ID Management
```python
def hybrid_expansion(self, input_data):
    # Use GraphBuilder for ID management
    prefix = GraphBuilder.alloc_prefix()
    
    # Load existing graph
    existing_graph = load_graph_from_file("template.json")
    
    # Add prefix to existing graph
    from comfy.graph_utils import add_graph_prefix
    prefixed_graph = add_graph_prefix(existing_graph, prefix)
    
    return {
        "result": ([f"{prefix}_output_node", 0],),
        "expand": prefixed_graph,
    }
```

### Advanced Patterns

#### Loop Implementation
```python
def loop_processor(self, input_data, iterations):
    graph = GraphBuilder()
    
    # Create initial node
    current_node = graph.node("InitialProcess", input=input_data)
    
    # Create loop nodes
    for i in range(iterations):
        current_node = graph.node("LoopProcess", 
                                input=current_node.out(0), 
                                iteration=i)
    
    return {
        "result": (current_node.out(0),),
        "expand": graph.finalize(),
    }
```

#### Conditional Subgraph
```python
def conditional_expansion(self, input_data, condition):
    graph = GraphBuilder()
    
    if condition:
        # Create path A
        node1 = graph.node("ProcessA", input=input_data)
        node2 = graph.node("ProcessA2", input=node1.out(0))
        result_node = node2
    else:
        # Create path B
        node1 = graph.node("ProcessB", input=input_data)
        node2 = graph.node("ProcessB2", input=node1.out(0))
        result_node = node2
    
    return {
        "result": (result_node.out(0),),
        "expand": graph.finalize(),
    }
```

#### Dynamic Workflow Generation
```python
def generate_workflow(self, workflow_config):
    graph = GraphBuilder()
    
    # Parse configuration
    steps = workflow_config.get("steps", [])
    
    # Create initial node
    current_node = graph.node("LoadImage", image=workflow_config["image"])
    
    # Add each step
    for step in steps:
        step_type = step["type"]
        step_params = step["params"]
        step_params["input"] = current_node.out(0)
        current_node = graph.node(step_type, **step_params)
    
    return {
        "result": (current_node.out(0),),
        "expand": graph.finalize(),
    }
```

### Caching Optimization

#### Efficient Subgraph Caching
```python
def optimized_expansion(self, model_link, image_link):
    graph = GraphBuilder()
    
    # Use rawLink for better caching
    model_input = graph.raw_link(model_link)
    image_input = graph.raw_link(image_link)
    
    # Create processing nodes
    process_node = graph.node("ProcessNode", 
                             model=model_input, 
                             image=image_input)
    
    return {
        "result": (process_node.out(0),),
        "expand": graph.finalize(),
    }
```

#### Input Declaration with rawLink
```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "model": ("MODEL", {"rawLink": True}),
            "image": ("IMAGE", {"rawLink": True}),
        }
    }
```

### Best Practices

#### ID Management
```python
def safe_expansion(self, input_data):
    # Always use GraphBuilder for ID management
    graph = GraphBuilder()
    
    # Create nodes with unique IDs
    node1 = graph.node("NodeType1", input=input_data)
    node2 = graph.node("NodeType2", input=node1.out(0))
    
    return {
        "result": (node2.out(0),),
        "expand": graph.finalize(),
    }
```

#### Error Handling
```python
def safe_expansion_with_error_handling(self, input_data):
    try:
        graph = GraphBuilder()
        
        # Create subgraph
        node1 = graph.node("NodeType1", input=input_data)
        node2 = graph.node("NodeType2", input=node1.out(0))
        
        return {
            "result": (node2.out(0),),
            "expand": graph.finalize(),
        }
    except Exception as e:
        # Fallback to simple processing
        return (simple_process(input_data),)
```

#### Performance Considerations
```python
def performance_optimized_expansion(self, input_data):
    # Use rawLink for inputs to improve caching
    graph = GraphBuilder()
    
    # Create efficient subgraph
    node1 = graph.node("EfficientNode1", input=input_data)
    node2 = graph.node("EfficientNode2", input=node1.out(0))
    
    return {
        "result": (node2.out(0),),
        "expand": graph.finalize(),
    }
```

### Common Use Cases

#### Workflow Templates
```python
def apply_workflow_template(self, image, template_name):
    graph = GraphBuilder()
    
    # Load template
    template = load_workflow_template(template_name)
    
    # Apply template to image
    current_node = graph.node("LoadImage", image=image)
    
    for step in template["steps"]:
        current_node = graph.node(step["type"], 
                                input=current_node.out(0), 
                                **step["params"])
    
    return {
        "result": (current_node.out(0),),
        "expand": graph.finalize(),
    }
```

#### Batch Processing
```python
def batch_processor(self, images, batch_size):
    graph = GraphBuilder()
    
    # Create batch processing nodes
    batch_node = graph.node("BatchProcess", 
                           images=images, 
                           batch_size=batch_size)
    
    return {
        "result": (batch_node.out(0),),
        "expand": graph.finalize(),
    }
```

## Data Lists and Sequential Processing

### Purpose
ComfyUI internally represents data as lists, normally length 1. When multiple data instances need processing, ComfyUI can process them sequentially to avoid memory issues or handle different data sizes.

### Length One Processing (Default)

#### Normal Operation
```python
# ComfyUI internally wraps outputs in lists
def process(self, input_data):
    result = process_data(input_data)
    return (result,)  # ComfyUI wraps this as [result]
```

#### Internal Representation
- **Output**: `(result,)` → ComfyUI wraps as `[result]`
- **Input**: Next node receives `result` (unwrapped)
- **Transparent**: You don't need to worry about this wrapping

### List Processing

#### Sequential Processing
When multiple data instances are processed:
- **Sequential execution**: Each item processed separately
- **Padding**: Shorter lists are padded by repeating last value
- **Output lists**: Same length as longest input list
- **Memory efficient**: Avoids VRAM issues with large batches

#### Example: Processing Multiple Images
```python
# Input: [image1, image2, image3]
# ComfyUI processes each image separately
# Output: [processed_image1, processed_image2, processed_image3]
```

### OUTPUT_IS_LIST

#### Purpose
Tells ComfyUI that returned lists should be treated as sequential data, not wrapped as single items.

#### Basic Usage
```python
class MyListProcessor:
    RETURN_TYPES = ("IMAGE", "STRING")
    OUTPUT_IS_LIST = (True, False)  # First output is list, second is not
    
    def process(self, input_data):
        results = []
        for item in input_data:
            processed = process_item(item)
            results.append(processed)
        
        return (results, "status")  # results is treated as list
```

#### Key Points
- **Tuple length**: Must match RETURN_TYPES length
- **Boolean values**: True for list outputs, False for single outputs
- **No wrapping**: Lists are not wrapped in additional lists
- **Sequential processing**: Enables downstream sequential processing

### INPUT_IS_LIST

#### Purpose
Allows nodes to receive entire lists in a single call instead of sequential processing.

#### Basic Usage
```python
class MyListHandler:
    INPUT_IS_LIST = True  # All inputs receive lists
    
    def process(self, images, batch_size):
        # images is a list of image batches
        # batch_size is a list (take first element: batch_size[0])
        batch_size = batch_size[0]
        
        # Process all images at once
        return (processed_images,)
```

#### Key Points
- **Node level**: Affects all inputs
- **List format**: All inputs come as lists
- **Widget values**: Use `widget[0]` to get actual value
- **Single call**: Process entire list in one execution

### Complete Example: ImageRebatch

```python
class ImageRebatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "rebatch"
    CATEGORY = "image/batch"
    
    def rebatch(self, images, batch_size):
        batch_size = batch_size[0]  # Get actual value from list
        
        output_list = []
        all_images = []
        
        # Flatten all image batches
        for img in images:  # Each img is a batch of images
            for i in range(img.shape[0]):  # Each i is a single image
                all_images.append(img[i:i+1])
        
        # Create new batches of requested size
        for i in range(0, len(all_images), batch_size):
            batch = torch.cat(all_images[i:i+batch_size], dim=0)
            output_list.append(batch)
        
        return (output_list,)
```

### Advanced Patterns

#### Mixed List Processing
```python
class MixedProcessor:
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    OUTPUT_IS_LIST = (True, False, True)  # Mixed list/single outputs
    
    def process(self, images, texts, numbers):
        processed_images = []
        status = "completed"
        processed_numbers = []
        
        for i, (img, text, num) in enumerate(zip(images, texts, numbers)):
            # Process each item
            processed_img = process_image(img)
            processed_images.append(processed_img)
            processed_numbers.append(num * 2)
        
        return (processed_images, status, processed_numbers)
```

#### Conditional List Processing
```python
class ConditionalListProcessor:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    
    def process(self, images, condition):
        condition = condition[0]  # Get actual boolean value
        
        if condition:
            # Process all images
            results = [process_image(img) for img in images]
        else:
            # Return original images
            results = images
        
        return (results,)
```

#### List Filtering
```python
class ListFilter:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    
    def process(self, images, filter_condition):
        filter_condition = filter_condition[0]
        
        if filter_condition:
            # Filter images based on some criteria
            filtered_images = [img for img in images if meets_criteria(img)]
        else:
            filtered_images = images
        
        return (filtered_images,)
```

### Best Practices

#### Memory Management
```python
class MemoryEfficientProcessor:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    
    def process(self, images, batch_size):
        batch_size = batch_size[0]
        results = []
        
        # Process in smaller batches to avoid memory issues
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            processed_batch = process_batch(batch)
            results.extend(processed_batch)
        
        return (results,)
```

#### Error Handling
```python
class RobustListProcessor:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    
    def process(self, images, fallback_image):
        fallback_image = fallback_image[0]
        results = []
        
        for img in images:
            try:
                processed = process_image(img)
                results.append(processed)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append(fallback_image)
        
        return (results,)
```

#### Performance Optimization
```python
class OptimizedListProcessor:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    
    def process(self, images, use_gpu):
        use_gpu = use_gpu[0]
        
        if use_gpu and torch.cuda.is_available():
            # Process on GPU
            results = [process_image_gpu(img) for img in images]
        else:
            # Process on CPU
            results = [process_image_cpu(img) for img in images]
        
        return (results,)
```

### Common Use Cases

#### Batch Size Management
```python
class BatchManager:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    
    def process(self, images, target_batch_size):
        target_batch_size = target_batch_size[0]
        results = []
        
        # Process images in target batch sizes
        for i in range(0, len(images), target_batch_size):
            batch = images[i:i+target_batch_size]
            processed_batch = process_batch(batch)
            results.append(processed_batch)
        
        return (results,)
```

#### Size Normalization
```python
class SizeNormalizer:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    
    def process(self, images, target_size):
        target_size = target_size[0]
        results = []
        
        for img in images:
            # Normalize to target size
            normalized = resize_image(img, target_size)
            results.append(normalized)
        
        return (results,)
```

#### Quality Filtering
```python
class QualityFilter:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    
    def process(self, images, quality_threshold):
        quality_threshold = quality_threshold[0]
        results = []
        
        for img in images:
            if get_image_quality(img) >= quality_threshold:
                results.append(img)
        
        return (results,)
```

## Annotated Examples - Practical Code Fragments

### Images and Masks

#### Load an Image
```python
# Load an image into a batch of size 1 (based on LoadImage source code)
from PIL import Image, ImageOps
import numpy as np
import torch

def load_image(image_path):
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)  # Handle EXIF orientation
    
    # Convert 16-bit grayscale to 8-bit
    if i.mode == 'I':
        i = i.point(lambda i: i * (1 / 255))
    
    # Convert to RGB
    image = i.convert("RGB")
    
    # Convert to numpy and normalize
    image = np.array(image).astype(np.float32) / 255.0
    
    # Convert to torch tensor and add batch dimension
    image = torch.from_numpy(image)[None,]  # [1, H, W, C]
    
    return image
```

#### Save an Image Batch
```python
# Save a batch of images (based on SaveImage source code)
from PIL import Image
import numpy as np

def save_image_batch(images, base_path):
    for (batch_number, image) in enumerate(images):
        # Convert tensor to numpy and scale to 0-255
        i = 255. * image.cpu().numpy()
        
        # Clip values and convert to uint8
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Create filepath with batch number
        filepath = f"{base_path}_{batch_number:04d}.png"
        img.save(filepath)
```

#### Invert a Mask
```python
# Inverting a mask (masks are normalized to [0,1])
def invert_mask(mask):
    return 1.0 - mask
```

#### Convert Mask to Image Shape
```python
# Convert mask to [B, H, W, C] format with C=1
def mask_to_image_shape(mask):
    if len(mask.shape) == 2:  # [H, W]
        mask = mask[None, :, :, None]  # [1, H, W, 1]
    elif len(mask.shape) == 3 and mask.shape[2] == 1:  # [H, W, C]
        mask = mask[None, :, :, :]  # [1, H, W, 1]
    elif len(mask.shape) == 3:  # [B, H, W]
        mask = mask[:, :, :, None]  # [B, H, W, 1]
    
    return mask
```

#### Using Masks as Transparency Layers
```python
# Create RGBA image using mask as transparency layer
def create_rgba_image(rgb_image, mask):
    # Invert mask back to original transparency layer
    mask = 1.0 - mask
    
    # Unsqueeze the C (channels) dimension
    mask = mask.unsqueeze(-1)
    
    # Concatenate along the C dimension
    rgba_image = torch.cat((rgb_image, mask), dim=-1)
    
    return rgba_image
```

### Noise

#### Creating Noise Variations
```python
# Create a noise object that mixes noise from two sources
class Noise_MixedNoise:
    def __init__(self, noise1, noise2, weight2):
        self.noise1 = noise1
        self.noise2 = noise2
        self.weight2 = weight2
    
    @property
    def seed(self):
        return self.noise1.seed
    
    def generate_noise(self, input_latent: torch.Tensor) -> torch.Tensor:
        noise1 = self.noise1.generate_noise(input_latent)
        noise2 = self.noise2.generate_noise(input_latent)
        return noise1 * (1.0 - self.weight2) + noise2 * self.weight2
```

### Advanced Image Processing Examples

#### Image Resizing
```python
def resize_image(image, target_size):
    # image: [B, H, W, C]
    # target_size: (width, height)
    
    # Convert to [B, C, H, W] for interpolation
    image_chw = image.permute(0, 3, 1, 2)
    
    # Resize using torch.nn.functional.interpolate
    resized = torch.nn.functional.interpolate(
        image_chw, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    
    # Convert back to [B, H, W, C]
    return resized.permute(0, 2, 3, 1)
```

#### Image Cropping
```python
def crop_image(image, x, y, width, height):
    # image: [B, H, W, C]
    # x, y: top-left corner
    # width, height: crop dimensions
    
    return image[:, y:y+height, x:x+width, :]
```

#### Image Padding
```python
def pad_image(image, padding):
    # image: [B, H, W, C]
    # padding: (top, bottom, left, right)
    
    return torch.nn.functional.pad(
        image, 
        (padding[2], padding[3], padding[0], padding[1]), 
        mode='constant', 
        value=0
    )
```

### Mask Processing Examples

#### Mask Thresholding
```python
def threshold_mask(mask, threshold=0.5):
    # Convert continuous mask to binary
    return (mask > threshold).float()
```

#### Mask Erosion
```python
def erode_mask(mask, kernel_size=3):
    # Simple erosion using max pooling
    # Convert to [B, C, H, W] for convolution
    mask_chw = mask.permute(0, 3, 1, 2)
    
    # Apply max pooling (erosion)
    eroded = torch.nn.functional.max_pool2d(
        mask_chw, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=kernel_size//2
    )
    
    # Convert back to [B, H, W, C]
    return eroded.permute(0, 2, 3, 1)
```

#### Mask Dilation
```python
def dilate_mask(mask, kernel_size=3):
    # Simple dilation using min pooling on inverted mask
    # Convert to [B, C, H, W] for convolution
    mask_chw = mask.permute(0, 3, 1, 2)
    
    # Invert mask for min pooling
    inverted = 1.0 - mask_chw
    
    # Apply min pooling (dilation on inverted)
    dilated_inverted = torch.nn.functional.max_pool2d(
        inverted, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=kernel_size//2
    )
    
    # Invert back
    dilated = 1.0 - dilated_inverted
    
    # Convert back to [B, H, W, C]
    return dilated.permute(0, 2, 3, 1)
```

### Tensor Manipulation Examples

#### Batch Operations
```python
def batch_mean(images):
    # images: [B, H, W, C]
    return torch.mean(images, dim=0, keepdim=True)  # [1, H, W, C]

def batch_std(images):
    # images: [B, H, W, C]
    return torch.std(images, dim=0, keepdim=True)  # [1, H, W, C]
```

#### Channel Operations
```python
def channel_mean(image):
    # image: [B, H, W, C]
    return torch.mean(image, dim=-1, keepdim=True)  # [B, H, W, 1]

def channel_std(image):
    # image: [B, H, W, C]
    return torch.std(image, dim=-1, keepdim=True)  # [B, H, W, 1]
```

#### Normalization
```python
def normalize_image(image, mean=0.5, std=0.5):
    # Normalize image to [0, 1] range
    return (image - mean) / std

def denormalize_image(image, mean=0.5, std=0.5):
    # Denormalize image from [0, 1] range
    return image * std + mean
```

### Color Space Conversions

#### RGB to Grayscale
```python
def rgb_to_grayscale(image):
    # image: [B, H, W, 3]
    # Standard RGB to grayscale conversion
    weights = torch.tensor([0.299, 0.587, 0.114], device=image.device)
    grayscale = torch.sum(image * weights, dim=-1, keepdim=True)
    return grayscale  # [B, H, W, 1]
```

#### Grayscale to RGB
```python
def grayscale_to_rgb(image):
    # image: [B, H, W, 1]
    # Replicate grayscale channel to RGB
    return image.repeat(1, 1, 1, 3)  # [B, H, W, 3]
```

### Memory Management Examples

#### Efficient Batch Processing
```python
def process_large_batch(images, batch_size=4):
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        # Process batch
        processed_batch = process_batch(batch)
        results.append(processed_batch)
        
        # Clear cache if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return torch.cat(results, dim=0)
```

#### Memory-Efficient Resizing
```python
def resize_large_image(image, target_size):
    # Process in chunks to avoid memory issues
    B, H, W, C = image.shape
    
    if H * W > 1024 * 1024:  # Large image
        # Process in smaller chunks
        chunk_size = 512
        chunks = []
        
        for y in range(0, H, chunk_size):
            for x in range(0, W, chunk_size):
                chunk = image[:, y:y+chunk_size, x:x+chunk_size, :]
                resized_chunk = resize_image(chunk, target_size)
                chunks.append(resized_chunk)
        
        # Reconstruct image
        return reconstruct_from_chunks(chunks, target_size)
    else:
        return resize_image(image, target_size)
```

### Error Handling Examples

#### Safe Image Loading
```python
def safe_load_image(image_path):
    try:
        image = load_image(image_path)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a default image
        return torch.zeros((1, 512, 512, 3))
```

#### Safe Tensor Operations
```python
def safe_tensor_operation(tensor, operation):
    try:
        result = operation(tensor)
        return result
    except Exception as e:
        print(f"Error in tensor operation: {e}")
        # Return original tensor or handle error
        return tensor
```

### Performance Optimization Examples

#### GPU Memory Management
```python
def gpu_optimized_processing(image):
    if torch.cuda.is_available():
        # Move to GPU
        image = image.cuda()
        
        # Process on GPU
        result = process_on_gpu(image)
        
        # Move back to CPU
        result = result.cpu()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return result
    else:
        return process_on_cpu(image)
```

#### Efficient Data Types
```python
def optimize_data_types(image):
    # Use appropriate data types for memory efficiency
    if image.dtype == torch.float64:
        image = image.float()  # Convert to float32
    
    # Use half precision if supported
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        image = image.half()
    
    return image
```

## Working with torch.Tensor

### Purpose
All core number crunching in ComfyUI is done by PyTorch. Understanding torch.Tensor is essential for manipulating images, latents, and masks in custom nodes.

### What is a Tensor?

#### Mathematical Definition
- **Tensor**: Mathematical generalization of vector or matrix to any number of dimensions
- **Rank**: Number of dimensions (vector=1, matrix=2)
- **Shape**: Size of each dimension
- **Example**: RGB image [H,W,3] → ComfyUI batch [B,H,W,C]

#### ComfyUI Tensor Conventions
```python
# ComfyUI tensor shapes
image_tensor.shape    # [B, H, W, C] - batch, height, width, channels
latent_tensor.shape   # [B, C, H, W] - batch, channels, height, width
mask_tensor.shape     # [B, H, W] or [H, W] - batch, height, width
```

### Tensor Manipulation

#### squeeze, unsqueeze, and reshape
```python
# Squeeze: Remove dimensions of size 1
tensor = torch.randn(1, 512, 512, 3)  # [1, H, W, C]
squeezed = tensor.squeeze(0)           # [H, W, C] - removes batch dimension

# Unsqueeze: Add dimensions of size 1
tensor = torch.randn(512, 512, 3)     # [H, W, C]
unsqueezed = tensor.unsqueeze(0)      # [1, H, W, C] - adds batch dimension

# Reshape: Change tensor shape
tensor = torch.randn(1, 512, 512, 3)  # [1, H, W, C]
reshaped = tensor.reshape(1, -1, 3)   # [1, H*W, C] - flatten height/width
```

#### Common Shape Operations
```python
# Add batch dimension
def add_batch_dim(tensor):
    if len(tensor.shape) == 3:  # [H, W, C]
        return tensor.unsqueeze(0)  # [1, H, W, C]
    return tensor

# Remove batch dimension
def remove_batch_dim(tensor):
    if len(tensor.shape) == 4) and tensor.shape[0] == 1:
        return tensor.squeeze(0)  # [H, W, C]
    return tensor

# Convert between channel formats
def channel_last_to_first(tensor):
    # [B, H, W, C] -> [B, C, H, W]
    return tensor.permute(0, 3, 1, 2)

def channel_first_to_last(tensor):
    # [B, C, H, W] -> [B, H, W, C]
    return tensor.permute(0, 2, 3, 1)
```

### Important Notation

#### Slice Notation
```python
# Basic slicing
tensor = torch.randn(2, 512, 512, 3)  # [B, H, W, C]

# Get first batch
first_batch = tensor[0, :, :, :]      # [H, W, C]

# Get specific region
region = tensor[:, 100:200, 100:200, :]  # [B, 100, 100, C]

# Get specific channel
red_channel = tensor[:, :, :, 0]      # [B, H, W]
```

#### Advanced Slicing
```python
# None for dimension insertion
tensor = torch.randn(512, 512, 3)    # [H, W, C]
expanded = tensor[:, None, :, :]     # [H, 1, W, C] - insert dimension

# Ellipsis for unspecified dimensions
tensor = torch.randn(2, 512, 512, 3)  # [B, H, W, C]
first_batch = tensor[0, ...]          # [H, W, C] - same as [0, :, :, :]

# Reshape with -1 for automatic calculation
tensor = torch.randn(2, 512, 512, 3)  # [B, H, W, C]
flattened = tensor.reshape(2, -1)     # [B, H*W*C] - flatten spatial and channel dims
```

### Elementwise Operations

#### Basic Operations
```python
# Elementwise arithmetic
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 2.0])

# Addition
result = a + b                        # tensor([4., 4.])

# Multiplication
result = a * b                        # tensor([3., 4.])

# Division
result = a / b                       # tensor([0.3333, 1.0000])

# Comparison
result = a == b                      # tensor([False, True])
result = a == 1                      # tensor([True, False])
```

#### Broadcasting
```python
# Scalar operations
tensor = torch.randn(2, 512, 512, 3)
scaled = tensor * 2.0                # Multiply all elements by 2
normalized = tensor / 255.0          # Normalize to [0, 1] range

# Shape broadcasting
a = torch.randn(2, 1, 1, 3)         # [B, 1, 1, C]
b = torch.randn(512, 512, 3)        # [H, W, C]
result = a + b                       # Broadcasts to [B, H, W, C]
```

### Tensor Truthiness

#### Boolean Operations
```python
# ❌ Wrong - ambiguous for multi-element tensors
tensor = torch.tensor([1.0, 2.0])
if tensor:  # RuntimeError: Boolean value ambiguous
    pass

# ✅ Correct - use .all() or .any()
tensor = torch.tensor([1.0, 2.0])
if tensor.all():                     # All elements are truthy
    pass
if tensor.any():                     # Any element is truthy
    pass

# ✅ Correct - check for None
if tensor is not None:               # Check if tensor exists
    pass
```

#### Practical Examples
```python
def safe_tensor_operation(tensor):
    # Check if tensor exists
    if tensor is not None:
        # Check if all values are valid
        if not torch.isnan(tensor).any():
            return process_tensor(tensor)
        else:
            print("Tensor contains NaN values")
    return None

def mask_validation(mask):
    # Check if mask is valid
    if mask is not None and mask.any():
        # Mask has some non-zero values
        return True
    return False
```

### Common Tensor Operations

#### Image Processing
```python
def normalize_image(image):
    # Normalize to [0, 1] range
    return (image - image.min()) / (image.max() - image.min())

def clamp_image(image, min_val=0.0, max_val=1.0):
    # Clamp values to range
    return torch.clamp(image, min_val, max_val)

def resize_tensor(tensor, target_size):
    # Resize tensor using interpolation
    return torch.nn.functional.interpolate(
        tensor, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
```

#### Mask Operations
```python
def threshold_mask(mask, threshold=0.5):
    # Convert to binary mask
    return (mask > threshold).float()

def invert_mask(mask):
    # Invert mask values
    return 1.0 - mask

def combine_masks(mask1, mask2, operation='add'):
    if operation == 'add':
        return torch.clamp(mask1 + mask2, 0, 1)
    elif operation == 'multiply':
        return mask1 * mask2
    elif operation == 'max':
        return torch.max(mask1, mask2)
```

#### Batch Operations
```python
def batch_mean(images):
    # Calculate mean across batch
    return torch.mean(images, dim=0, keepdim=True)

def batch_std(images):
    # Calculate standard deviation across batch
    return torch.std(images, dim=0, keepdim=True)

def batch_concat(images):
    # Concatenate images along batch dimension
    return torch.cat(images, dim=0)
```

### Memory Management

#### Efficient Tensor Operations
```python
def memory_efficient_processing(tensor):
    # Use in-place operations when possible
    tensor.mul_(0.5)  # In-place multiplication
    
    # Clear cache if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return tensor

def chunk_processing(tensor, chunk_size=512):
    # Process large tensors in chunks
    results = []
    for i in range(0, tensor.shape[0], chunk_size):
        chunk = tensor[i:i+chunk_size]
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    
    return torch.cat(results, dim=0)
```

#### Data Type Optimization
```python
def optimize_tensor_dtype(tensor):
    # Use appropriate data types
    if tensor.dtype == torch.float64:
        tensor = tensor.float()  # Convert to float32
    
    # Use half precision if supported
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        tensor = tensor.half()
    
    return tensor
```

### Error Handling

#### Safe Tensor Operations
```python
def safe_tensor_operation(tensor, operation):
    try:
        # Check tensor validity
        if tensor is None:
            raise ValueError("Tensor is None")
        
        if torch.isnan(tensor).any():
            raise ValueError("Tensor contains NaN values")
        
        # Perform operation
        result = operation(tensor)
        return result
        
    except Exception as e:
        print(f"Error in tensor operation: {e}")
        return None

def validate_tensor_shape(tensor, expected_shape):
    if tensor.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {tensor.shape}")
    return True
```

### Performance Tips

#### GPU Operations
```python
def gpu_optimized_processing(tensor):
    if torch.cuda.is_available():
        # Move to GPU
        tensor = tensor.cuda()
        
        # Process on GPU
        result = process_on_gpu(tensor)
        
        # Move back to CPU
        result = result.cpu()
        
        return result
    else:
        return process_on_cpu(tensor)
```

#### Efficient Broadcasting
```python
def efficient_broadcasting(tensor1, tensor2):
    # Ensure compatible shapes
    if tensor1.shape != tensor2.shape:
        # Broadcast to compatible shapes
        tensor1 = tensor1.unsqueeze(-1)  # Add dimension
        tensor2 = tensor2.unsqueeze(0)   # Add dimension
    
    return tensor1 + tensor2
```

## Working with the UI - JavaScript Extensions

### Purpose
ComfyUI can be extended through JavaScript to modify the client interface, add custom widgets, and enhance user experience.

### Extending the Comfy Client

#### Three-Step Process
1. **Export WEB_DIRECTORY** from Python module
2. **Place .js files** in the directory
3. **Register extension** using `app.registerExtension`

### Step 1: Exporting WEB_DIRECTORY

#### Python Module Setup
```python
# __init__.py
from .my_node import MyCustomNode

NODE_CLASS_MAPPINGS = {
    "My Custom Node": MyCustomNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "My Custom Node": "🎯 My Custom Node",
}

WEB_DIRECTORY = "./js"  # JavaScript files directory

__all__ = [
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "WEB_DIRECTORY"
]
```

#### Directory Structure
```
my_custom_node/
├── __init__.py              # Python module entry point
├── my_node.py              # Node implementations
├── requirements.txt         # Dependencies
└── js/                     # JavaScript extensions
    ├── my_extension.js     # Main extension file
    └── widgets.js          # Custom widgets
```

### Step 2: Including .js Files

#### File Loading
- **Automatic loading**: All .js files in WEB_DIRECTORY are loaded
- **No specification needed**: Files are loaded automatically
- **Order matters**: Files are loaded in alphabetical order
- **Only .js files**: Other resources must be added programmatically

#### Resource Access
```javascript
// Access CSS files programmatically
const cssPath = "extensions/custom_node_subfolder/style.css";
const link = document.createElement("link");
link.rel = "stylesheet";
link.href = cssPath;
document.head.appendChild(link);
```

### Step 3: Registering an Extension

#### Basic Extension Structure
```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "my.unique.extension.name",
    async setup() {
        console.log("Extension setup complete!");
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Called before node is registered
    },
    async afterRegisterNodeDef(nodeType, nodeData, app) {
        // Called after node is registered
    }
});
```

#### Complete Example
```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "comfyui.custom.extension",
    
    async setup() {
        console.log("Custom extension loaded!");
        this.addCustomStyles();
        this.setupEventListeners();
    },
    
    addCustomStyles() {
        const style = document.createElement("style");
        style.textContent = `
            .custom-widget {
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
        `;
        document.head.appendChild(style);
    },
    
    setupEventListeners() {
        // Add custom event listeners
        document.addEventListener("customEvent", (event) => {
            console.log("Custom event received:", event.detail);
        });
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType === "MyCustomNode") {
            // Modify node before registration
            nodeData.category = "Custom/My Category";
        }
    },
    
    async afterRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType === "MyCustomNode") {
            // Add custom functionality after registration
            this.addCustomWidget(nodeType, nodeData);
        }
    },
    
    addCustomWidget(nodeType, nodeData) {
        // Add custom widget functionality
        console.log(`Added custom widget for ${nodeType}`);
    }
});
```

### Available Hooks

#### Extension Lifecycle Hooks
```javascript
app.registerExtension({
    name: "my.extension",
    
    // Called when extension is loaded
    async setup() {
        console.log("Extension setup");
    },
    
    // Called before ComfyUI starts
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Modify node before registration
    },
    
    // Called after ComfyUI starts
    async afterRegisterNodeDef(nodeType, nodeData, app) {
        // Add functionality after registration
    },
    
    // Called when workflow is executed
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Pre-execution setup
    },
    
    // Called after workflow execution
    async afterRegisterNodeDef(nodeType, nodeData, app) {
        // Post-execution cleanup
    }
});
```

#### Node-Specific Hooks
```javascript
app.registerExtension({
    name: "node.customizer",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType === "MyCustomNode") {
            // Customize node appearance
            nodeData.category = "Custom/My Category";
            nodeData.color = "#ff6b6b";
            nodeData.title = "🎯 My Custom Node";
        }
    },
    
    async afterRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType === "MyCustomNode") {
            // Add custom widgets
            this.addCustomControls(nodeType, nodeData);
        }
    },
    
    addCustomControls(nodeType, nodeData) {
        // Add custom UI controls
        const controls = document.createElement("div");
        controls.className = "custom-controls";
        controls.innerHTML = `
            <button onclick="customFunction()">Custom Action</button>
        `;
        // Add to node interface
    }
});
```

### Custom Widgets

#### Creating Custom Widgets
```javascript
app.registerExtension({
    name: "custom.widgets",
    
    async setup() {
        this.addCustomWidgets();
    },
    
    addCustomWidgets() {
        // Add custom widget types
        LiteGraph.registerNodeType("CustomWidget", {
            title: "Custom Widget",
            category: "Custom",
            
            onAdded(graph) {
                this.addInput("input", "number");
                this.addOutput("output", "number");
            },
            
            onExecute() {
                // Custom widget logic
                const input = this.getInputData(0);
                this.setOutputData(0, input * 2);
            }
        });
    }
});
```

#### Interactive Widgets
```javascript
app.registerExtension({
    name: "interactive.widgets",
    
    async setup() {
        this.createInteractiveWidgets();
    },
    
    createInteractiveWidgets() {
        // Create interactive elements
        const widget = document.createElement("div");
        widget.className = "interactive-widget";
        widget.innerHTML = `
            <input type="range" min="0" max="100" value="50" id="slider">
            <span id="value">50</span>
        `;
        
        const slider = widget.querySelector("#slider");
        const value = widget.querySelector("#value");
        
        slider.addEventListener("input", (e) => {
            value.textContent = e.target.value;
            // Trigger custom event
            document.dispatchEvent(new CustomEvent("sliderChanged", {
                detail: { value: e.target.value }
            }));
        });
        
        // Add to ComfyUI interface
        document.body.appendChild(widget);
    }
});
```

### Event Handling

#### Custom Events
```javascript
app.registerExtension({
    name: "event.handler",
    
    async setup() {
        this.setupEventHandling();
    },
    
    setupEventHandling() {
        // Listen for custom events
        document.addEventListener("customEvent", (event) => {
            console.log("Custom event received:", event.detail);
            this.handleCustomEvent(event.detail);
        });
        
        // Listen for ComfyUI events
        app.addEventListener("nodeAdded", (event) => {
            console.log("Node added:", event.detail);
        });
        
        app.addEventListener("workflowExecuted", (event) => {
            console.log("Workflow executed:", event.detail);
        });
    },
    
    handleCustomEvent(detail) {
        // Handle custom event
        console.log("Handling custom event:", detail);
    }
});
```

#### Node Events
```javascript
app.registerExtension({
    name: "node.events",
    
    async setup() {
        this.setupNodeEvents();
    },
    
    setupNodeEvents() {
        // Listen for node-specific events
        document.addEventListener("nodeSelected", (event) => {
            const node = event.detail;
            this.highlightNode(node);
        });
        
        document.addEventListener("nodeDeselected", (event) => {
            const node = event.detail;
            this.unhighlightNode(node);
        });
    },
    
    highlightNode(node) {
        // Add highlighting to node
        node.addClass("highlighted");
    },
    
    unhighlightNode(node) {
        // Remove highlighting from node
        node.removeClass("highlighted");
    }
});
```

### Advanced Features

#### Custom Node Categories
```javascript
app.registerExtension({
    name: "custom.categories",
    
    async setup() {
        this.addCustomCategories();
    },
    
    addCustomCategories() {
        // Add custom category styles
        const style = document.createElement("style");
        style.textContent = `
            .litegraph .lgraphcanvas .node[data-type="MyCustomNode"] {
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                border: 2px solid #ff6b6b;
                border-radius: 10px;
            }
            
            .litegraph .lgraphcanvas .node[data-type="MyCustomNode"]:hover {
                box-shadow: 0 0 20px rgba(255, 107, 107, 0.5);
            }
        `;
        document.head.appendChild(style);
    }
});
```

#### Dynamic UI Updates
```javascript
app.registerExtension({
    name: "dynamic.ui",
    
    async setup() {
        this.setupDynamicUpdates();
    },
    
    setupDynamicUpdates() {
        // Update UI based on node changes
        setInterval(() => {
            this.updateNodeStatus();
        }, 1000);
    },
    
    updateNodeStatus() {
        // Update node status indicators
        const nodes = document.querySelectorAll(".node");
        nodes.forEach(node => {
            const status = this.getNodeStatus(node);
            this.updateNodeIndicator(node, status);
        });
    },
    
    getNodeStatus(node) {
        // Determine node status
        return "active"; // or "inactive", "error", etc.
    },
    
    updateNodeIndicator(node, status) {
        // Update visual indicator
        node.setAttribute("data-status", status);
    }
});
```

### Best Practices

#### Error Handling
```javascript
app.registerExtension({
    name: "robust.extension",
    
    async setup() {
        try {
            this.initializeExtension();
        } catch (error) {
            console.error("Extension initialization failed:", error);
        }
    },
    
    initializeExtension() {
        // Extension initialization code
        console.log("Extension initialized successfully");
    }
});
```

#### Performance Optimization
```javascript
app.registerExtension({
    name: "optimized.extension",
    
    async setup() {
        this.setupOptimizedExtension();
    },
    
    setupOptimizedExtension() {
        // Use debouncing for frequent events
        this.debouncedUpdate = this.debounce(this.updateUI, 100);
        
        // Use event delegation for efficiency
        document.addEventListener("click", (event) => {
            if (event.target.matches(".custom-button")) {
                this.handleButtonClick(event.target);
            }
        });
    },
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
});
```

## ComfyUI Hooks - Extension Lifecycle

### Purpose
ComfyUI provides a comprehensive hook system that allows extensions to modify client behavior at various points during execution. Understanding these hooks is essential for creating powerful, well-integrated extensions.

### Extension Hooks Overview

#### Hook Invocation
- **Synchronous**: `#invokeExtensions(method, ...args)`
- **Asynchronous**: `#invokeExtensionsAsync(method, ...args)`
- **All extensions**: Hooks are called on all registered extensions
- **Method presence**: Only called if the method exists on the extension

### Commonly Used Hooks

#### beforeRegisterNodeDef()
```javascript
async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // Called once for each node type
    // nodeType: Template for all nodes of this type
    // nodeData: Node definition from Python code
    // app: Main Comfy app object
}
```

**Purpose**: Modify node behavior before registration
**Usage**: Most common hook for node customization
**Parameters**:
- `nodeType`: Node template (modify prototype for all instances)
- `nodeData`: Node metadata from Python
- `app`: ComfyUI app reference

#### Method Hijacking Pattern
```javascript
async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeType.comfyClass === "MyNodeClass") {
        // Store original method
        const originalMethod = nodeType.prototype.onConnectionsChange;
        
        // Replace with custom implementation
        nodeType.prototype.onConnectionsChange = function(side, slot, connect, link_info, output) {
            // Call original method first
            const result = originalMethod?.apply(this, arguments);
            
            // Add custom behavior
            console.log("Connection changed!");
            
            return result;
        };
    }
}
```

**Key Points**:
- **Check nodeType.comfyClass**: Target specific node types
- **Store original method**: Preserve existing functionality
- **Use ?.apply**: Safe method calling
- **Return result**: Maintain method contract

#### nodeCreated()
```javascript
async nodeCreated(node) {
    // Called when specific node instance is created
    // node: Individual node instance
}
```

**Purpose**: Modify individual node instances
**Usage**: Instance-specific customizations
**Parameters**:
- `node`: Specific node instance

**Example**:
```javascript
async nodeCreated(node) {
    if (node.comfyClass === "MyNodeClass") {
        // Add instance-specific properties
        node.customProperty = "value";
        
        // Add event listeners
        node.addEventListener("click", () => {
            console.log("Node clicked!");
        });
    }
}
```

#### init()
```javascript
async init() {
    // Called when ComfyUI webpage loads
    // Before any nodes are registered
    // After graph object creation
}
```

**Purpose**: Modify core ComfyUI behavior
**Usage**: Global modifications, core hijacking
**Warning**: High impact, potential compatibility issues

**Example**:
```javascript
async init() {
    // Hijack core app methods
    const originalMethod = app.someCoreMethod;
    app.someCoreMethod = function(...args) {
        console.log("Core method called");
        return originalMethod.apply(this, args);
    };
}
```

#### setup()
```javascript
async setup() {
    // Called at end of startup process
    // Good for event listeners and menu additions
}
```

**Purpose**: Final setup after all initialization
**Usage**: Event listeners, menu additions, global setup
**Note**: Use `afterConfigureGraph` for workflow-specific setup

**Example**:
```javascript
async setup() {
    // Add global event listeners
    document.addEventListener("keydown", (event) => {
        if (event.ctrlKey && event.key === "s") {
            this.saveWorkflow();
        }
    });
    
    // Add to global menus
    this.addToMainMenu();
}
```

### Hook Call Sequences

#### Web Page Load Sequence
```
invokeExtensionsAsync init
invokeExtensionsAsync addCustomNodeDefs
invokeExtensionsAsync getCustomWidgets
invokeExtensionsAsync beforeRegisterNodeDef    [repeated multiple times]
invokeExtensionsAsync registerCustomNodes
invokeExtensionsAsync beforeConfigureGraph
invokeExtensionsAsync nodeCreated
invokeExtensions      loadedGraphNode
invokeExtensionsAsync afterConfigureGraph
invokeExtensionsAsync setup
```

#### Loading Workflow Sequence
```
invokeExtensionsAsync beforeConfigureGraph
invokeExtensionsAsync beforeRegisterNodeDef   [zero, one, or multiple times]
invokeExtensionsAsync nodeCreated             [repeated multiple times]
invokeExtensions      loadedGraphNode         [repeated multiple times]
invokeExtensionsAsync afterConfigureGraph
```

#### Adding New Node Sequence
```
invokeExtensionsAsync nodeCreated
```

### Advanced Hook Patterns

#### Conditional Node Modification
```javascript
async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // Check if this is our custom node
    if (nodeType.comfyClass === "MyCustomNode") {
        // Add custom properties
        nodeType.prototype.customMethod = function() {
            console.log("Custom method called");
        };
        
        // Modify existing methods
        this.hijackNodeMethod(nodeType, "onExecute");
    }
}

hijackNodeMethod(nodeType, methodName) {
    const originalMethod = nodeType.prototype[methodName];
    nodeType.prototype[methodName] = function(...args) {
        // Pre-execution logic
        console.log(`Before ${methodName}`);
        
        // Call original method
        const result = originalMethod?.apply(this, args);
        
        // Post-execution logic
        console.log(`After ${methodName}`);
        
        return result;
    };
}
```

#### Inter-Extension Communication
```javascript
async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeType.comfyClass === "MyNodeClass") {
        // Check if other extensions have modified this node
        if (nodeType.prototype._extensionModified) {
            console.log("Node already modified by another extension");
        }
        
        // Mark as modified
        nodeType.prototype._extensionModified = true;
        
        // Add extension-specific behavior
        this.addExtensionBehavior(nodeType);
    }
}
```

#### Safe Method Hijacking
```javascript
async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeType.comfyClass === "MyNodeClass") {
        this.safeHijackMethod(nodeType, "onConnectionsChange", (originalMethod) => {
            return function(side, slot, connect, link_info, output) {
                // Pre-hook logic
                console.log("Before connection change");
                
                // Call original method
                const result = originalMethod?.apply(this, arguments);
                
                // Post-hook logic
                console.log("After connection change");
                
                return result;
            };
        });
    }
}

safeHijackMethod(nodeType, methodName, newMethodFactory) {
    const originalMethod = nodeType.prototype[methodName];
    
    // Create new method
    const newMethod = newMethodFactory(originalMethod);
    
    // Store original for potential restoration
    newMethod._originalMethod = originalMethod;
    
    // Replace method
    nodeType.prototype[methodName] = newMethod;
}
```

### Best Practices

#### Hook Implementation Guidelines
```javascript
app.registerExtension({
    name: "best.practices.extension",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 1. Always check if this is the right node type
        if (nodeType.comfyClass !== "MyNodeClass") {
            return;
        }
        
        // 2. Store original methods safely
        const originalMethod = nodeType.prototype.onExecute;
        
        // 3. Use defensive programming
        if (typeof originalMethod !== "function") {
            console.warn("Original method not found");
            return;
        }
        
        // 4. Implement new method
        nodeType.prototype.onExecute = function(...args) {
            try {
                // Pre-execution logic
                this.preExecute();
                
                // Call original method
                const result = originalMethod.apply(this, args);
                
                // Post-execution logic
                this.postExecute();
                
                return result;
            } catch (error) {
                console.error("Error in hijacked method:", error);
                throw error;
            }
        };
    },
    
    async nodeCreated(node) {
        // 1. Check node type
        if (node.comfyClass !== "MyNodeClass") {
            return;
        }
        
        // 2. Add instance-specific behavior
        this.setupNodeInstance(node);
    },
    
    setupNodeInstance(node) {
        // Add instance properties
        node.extensionData = {
            created: Date.now(),
            modified: false
        };
        
        // Add event listeners
        node.addEventListener("customEvent", (event) => {
            this.handleNodeEvent(node, event);
        });
    }
});
```

#### Error Handling
```javascript
async beforeRegisterNodeDef(nodeType, nodeData, app) {
    try {
        if (nodeType.comfyClass === "MyNodeClass") {
            this.modifyNode(nodeType);
        }
    } catch (error) {
        console.error("Error in beforeRegisterNodeDef:", error);
        // Don't throw - let other extensions continue
    }
}

modifyNode(nodeType) {
    // Safe method modification
    const originalMethod = nodeType.prototype.onExecute;
    
    if (originalMethod) {
        nodeType.prototype.onExecute = function(...args) {
            try {
                return originalMethod.apply(this, args);
            } catch (error) {
                console.error("Error in modified method:", error);
                // Handle error gracefully
                return null;
            }
        };
    }
}
```

#### Performance Considerations
```javascript
async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // 1. Early return for irrelevant nodes
    if (nodeType.comfyClass !== "MyNodeClass") {
        return;
    }
    
    // 2. Cache expensive operations
    if (!this._cachedData) {
        this._cachedData = this.computeExpensiveData();
    }
    
    // 3. Use efficient method hijacking
    this.efficientHijack(nodeType);
}

efficientHijack(nodeType) {
    // Use arrow functions for better performance
    const originalMethod = nodeType.prototype.onExecute;
    nodeType.prototype.onExecute = (...args) => {
        // Efficient implementation
        return originalMethod?.apply(this, args);
    };
}
```

## ComfyUI Objects - Core Architecture

### Purpose
Understanding ComfyUI's core objects is essential for creating sophisticated extensions and custom nodes. This section covers the main objects and their properties.

### LiteGraph Foundation

#### Overview
- **Base framework**: ComfyUI is built on top of LiteGraph
- **Documentation**: Available at `doc/index.html` in LiteGraph repository
- **Core functionality**: Much of ComfyUI's functionality comes from LiteGraph
- **Repository**: https://github.com/jagenjo/litegraph.js

### ComfyApp Object

#### Access and Properties
```javascript
import { app } from "../../scripts/app.js";

// Key properties
app.canvas          // LGraphCanvas object (UI)
app.canvasEl        // DOM <canvas> element
app.graph           // LGraph object (logical graph)
app.runningNodeId   // Currently executing node
app.ui              // UI elements (queue, menu, dialogs)
```

#### Important Functions
```javascript
// Core functions
app.graphToPrompt()     // Convert graph to prompt
app.loadGraphData()     // Load a graph
app.queuePrompt()       // Submit prompt to queue
app.registerExtension() // Add extension
```

#### Usage Examples
```javascript
// Get current graph
const graph = app.graph;

// Get specific node
const node = app.graph._nodes_by_id(nodeId);

// Convert graph to prompt
const prompt = app.graphToPrompt();

// Submit workflow
app.queuePrompt(prompt);
```

### LGraph Object

#### Purpose
- **Logical representation**: Current state of graph (nodes and links)
- **Manipulation**: Use LiteGraph functions for graph operations
- **Documentation**: Refer to LiteGraph documentation for functions

#### Node and Link Access
```javascript
// Get node by ID
const node = app.graph._nodes_by_id(nodeId);

// Iterate through node inputs
node.inputs.forEach(input => {
    const linkId = input.link;
    if (linkId) {
        const link = app.graph.links[linkId];
        const upstreamNodeId = link.origin_id;
        const upstreamSlot = link.origin_slot;
        const downstreamSlot = link.target_slot;
        const dataType = link.type;
    }
});

// Get all links
const allLinks = app.graph.links;
```

### LLink Object

#### Properties
```javascript
const link = app.graph.links[linkId];

// Link properties
link.id           // Unique link ID
link.origin_id    // Upstream node ID
link.origin_slot  // Upstream output slot
link.target_id    // Downstream node ID
link.target_slot  // Downstream input slot
link.type         // Data type string
```

#### Usage
```javascript
// Create links using LiteGraph functions
node.connect(outputSlot, targetNode, inputSlot);

// Avoid creating LLink objects directly
// Use LiteGraph functions instead
```

### ComfyNode Object

#### Properties
```javascript
const node = app.graph._nodes_by_id(nodeId);

// Core properties
node.bgcolor           // Background color
node.comfyClass        // Python class name
node.flags             // Node state flags
node.graph             // Reference to LGraph
node.id                // Unique node ID
node.input_type        // List of input types
node.inputs            // List of inputs
node.mode              // Node mode (0=normal, 2=muted, 4=bypassed)
node.order             // Execution order
node.pos               // [x, y] position
node.properties        // Node properties
node.properties_info   // Property types and defaults
node.size              // [width, height] dimensions
node.title             // Display title
node.type              // Unique node class name
node.widgets           // List of widgets
node.widgets_values    // Current widget values
```

#### Important Functions

##### Inputs and Outputs
```javascript
// Add inputs
node.addInput(name, type);
node.addInputs(inputArray);

// Find inputs
const slotIndex = node.findInputSlot(name);
const inputSlot = node.findInputSlotByType(type);

// Remove inputs
node.removeInput(slotIndex);

// Get connected nodes
const inputNode = node.getInputNode(inputSlot);
const outputNodes = node.getOutputNodes(outputSlot);
const inputLink = node.getInputLink(inputSlot);
```

##### Widgets
```javascript
// Add widgets
node.addWidget(type, name, value, options);
node.addCustomWidget(widget);
node.addDOMWidget(element);

// Convert widget to input
node.convertWidgetToInput(widgetName);
```

##### Connections
```javascript
// Connect nodes
node.connect(outputSlot, targetNode, inputSlot);
node.connectByType(outputSlot, targetNode, inputType);
node.connectByTypeOutput(inputSlot, targetNode, inputType);

// Disconnect
node.disconnectInput(inputName);
node.disconnectOutput(outputSlot, targetNode, inputSlot);

// Connection events
node.onConnectionChange(side, slot, connect, linkInfo, output);
node.onConnectInput(inputSlot, outputNode, outputSlot);
```

##### Display
```javascript
// Redraw node
node.setDirtyCanvas(foreground, background);

// Custom drawing
node.onDrawBackground(ctx);  // Background drawing
node.onDrawForeground(ctx);  // Foreground drawing

// Node title
node.getTitle();

// Collapse state
node.collapse(boolean);
```

##### Other Functions
```javascript
// Change node mode
node.changeMode(mode);  // 0=normal, 2=muted, 4=bypassed
```

### Inputs and Widgets

#### Input Structure
```javascript
const input = node.inputs[inputIndex];

// Input properties
input.name    // Input name
input.type    // Input type
input.link    // Connected LLink reference
input.widget  // Reference to converted widget (if applicable)
```

#### Widget Structure
```javascript
const widget = node.widgets[widgetIndex];

// Widget properties
widget.callback  // Value change callback
widget.last_y    // Vertical position
widget.name      // Widget name
widget.options   // Widget options
widget.type      // Widget type
widget.value     // Current value
```

#### Widget Types
```javascript
// Built-in widget types
app.widgets.BOOLEAN     // Boolean widget
app.widgets.INT         // Integer widget
app.widgets.FLOAT       // Float widget
app.widgets.STRING      // String widget
app.widgets.COMBO       // Dropdown widget
app.widgets.IMAGEUPLOAD // Image upload widget

// Custom widget types
// Add via getCustomWidgets hook
```

#### Linked Widgets
```javascript
// Linked widget example
const linkedWidget = {
    type: 'int:seed',  // base_widget_type:base_widget_name
    name: 'control_after_generate',
    value: 42
};
```

### Prompt Object

#### Structure
```javascript
const prompt = app.graphToPrompt();

// Prompt properties
prompt.output    // Node data mapping
prompt.workflow  // Workflow structure
```

#### Output Object
```javascript
// Node data structure
prompt.output[nodeId] = {
    class_type: "NodeClassName",
    inputs: {
        inputName: "value",           // Widget value
        inputName: [nodeId, slot],    // Connected input
        inputName: undefined          // Unconnected converted input
    }
};
```

#### Workflow Object
```javascript
prompt.workflow = {
    config: {},                    // Configuration options
    extra: {
        ds: {},                    // View description
        groups: [],                // Workflow groups
        last_link_id: 123,         // Last link ID
        last_node_id: 456          // Last node ID
    },
    links: [                       // Link array
        [linkId, originId, originSlot, targetId, targetSlot, type]
    ],
    nodes: [                       // Node array
        {
            flags: {},
            id: 123,
            inputs: [],
            mode: 0,
            order: 1,
            pos: [100, 200],
            properties: {},
            size: [200, 100],
            type: "NodeType",
            widgets_values: [],
            outputs: [             // If node has outputs
                {
                    name: "output",
                    type: "IMAGE",
                    links: [1, 2, 3],
                    shape: 3,
                    slot_index: 0
                }
            ]
        }
    ],
    version: 0.4                   // LiteGraph version
};
```

### Advanced Usage Examples

#### Node Manipulation
```javascript
// Create custom node behavior
async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeType.comfyClass === "MyCustomNode") {
        // Add custom methods
        nodeType.prototype.customMethod = function() {
            console.log("Custom method called");
        };
        
        // Hijack existing methods
        const originalOnExecute = nodeType.prototype.onExecute;
        nodeType.prototype.onExecute = function() {
            console.log("Before execution");
            const result = originalOnExecute?.apply(this, arguments);
            console.log("After execution");
            return result;
        };
    }
}
```

#### Graph Traversal
```javascript
// Traverse graph connections
function traverseGraph(startNodeId) {
    const visited = new Set();
    const queue = [startNodeId];
    
    while (queue.length > 0) {
        const nodeId = queue.shift();
        if (visited.has(nodeId)) continue;
        
        visited.add(nodeId);
        const node = app.graph._nodes_by_id(nodeId);
        
        // Process node
        console.log(`Processing node: ${node.title}`);
        
        // Add connected nodes to queue
        node.inputs.forEach(input => {
            if (input.link) {
                const link = app.graph.links[input.link];
                queue.push(link.origin_id);
            }
        });
    }
}
```

#### Custom Widget Creation
```javascript
// Add custom widget type
app.registerExtension({
    name: "custom.widgets",
    
    getCustomWidgets(nodeType, nodeData, app) {
        return {
            CUSTOM_SLIDER: {
                name: "custom_slider",
                draw: function(ctx, node, widgetWidth, y, widget) {
                    // Custom drawing code
                },
                mouse: function(event, pos, node) {
                    // Mouse interaction code
                }
            }
        };
    }
});
```

### Best Practices

#### Object Access
```javascript
// Safe object access
function safeGetNode(nodeId) {
    try {
        return app.graph._nodes_by_id(nodeId);
    } catch (error) {
        console.error("Error accessing node:", error);
        return null;
    }
}

// Validate node properties
function validateNode(node) {
    if (!node || !node.comfyClass) {
        console.warn("Invalid node object");
        return false;
    }
    return true;
}
```

#### Performance Optimization
```javascript
// Cache frequently accessed objects
const graph = app.graph;
const canvas = app.canvas;

// Use efficient traversal
function getConnectedNodes(nodeId) {
    const node = app.graph._nodes_by_id(nodeId);
    const connected = [];
    
    node.inputs.forEach(input => {
        if (input.link) {
            const link = app.graph.links[input.link];
            connected.push(link.origin_id);
        }
    });
    
    return connected;
}
```

## ComfyUI Settings - Extension Configuration

### Purpose
ComfyUI provides a powerful settings system that allows extensions to add configuration options that appear in the ComfyUI settings panel. This enables users to customize extension behavior without modifying code.

### Basic Operation

#### Adding a Setting
```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "My Extension",
    settings: [
        {
            id: "example.boolean",
            name: "Example boolean setting",
            type: "boolean",
            defaultValue: false,
        },
    ],
});
```

**Key Points**:
- **Unique ID**: Must be unique across all extensions
- **Category**: ID split by `.` determines category placement
- **No dots**: Appears in "Other" category
- **One dot**: Left part = category, right part = section
- **Multiple dots**: Only first two parts used

#### Reading a Setting
```javascript
import { app } from "../../scripts/app.js";

if (app.extensionManager.setting.get('example.boolean')) {
    console.log("Setting is enabled.");
} else {
    console.log("Setting is disabled.");
}
```

#### Reacting to Changes
```javascript
{
    id: "example.boolean",
    name: "Example boolean setting",
    type: "boolean",
    defaultValue: false,
    onChange: (newVal, oldVal) => {
        console.log(`Setting was changed from ${oldVal} to ${newVal}`);
    },
}
```

**Note**: `onChange` is called on every page load when extension is registered.

#### Writing a Setting
```javascript
import { app } from "../../scripts/app.js";

try {
    await app.extensionManager.setting.set("example.boolean", true);
} catch (error) {
    console.error(`Error changing setting: ${error}`);
}
```

### Setting Types

#### Boolean
```javascript
{
    id: "example.boolean",
    name: "Example boolean setting",
    type: "boolean",
    defaultValue: false,
    onChange: (newVal, oldVal) => {
        console.log(`Setting was changed from ${oldVal} to ${newVal}`);
    },
}
```

**Features**:
- **Toggle switch**: On/off toggle
- **PrimeVue component**: ToggleSwitch
- **Use cases**: Enable/disable features, show/hide elements

#### Text
```javascript
{
    id: "example.text",
    name: "Example text setting",
    type: "text",
    defaultValue: "Foo",
    onChange: (newVal, oldVal) => {
        console.log(`Setting was changed from ${oldVal} to ${newVal}`);
    },
}
```

**Features**:
- **Freeform text**: User can type anything
- **PrimeVue component**: InputText
- **Use cases**: File paths, custom strings, configuration values

#### Number
```javascript
{
    id: "example.number",
    name: "Example number setting",
    type: "number",
    defaultValue: 42,
    attrs: {
        showButtons: true,
        maxFractionDigits: 1,
    },
    onChange: (newVal, oldVal) => {
        console.log(`Setting was changed from ${oldVal} to ${newVal}`);
    },
}
```

**Features**:
- **Numeric input**: Numbers only
- **PrimeVue component**: InputNumber
- **Attributes**: `showButtons`, `maxFractionDigits`
- **Use cases**: Counts, dimensions, thresholds

#### Slider
```javascript
{
    id: "example.slider",
    name: "Example slider setting",
    type: "slider",
    attrs: {
        min: -10,
        max: 10,
        step: 0.5,
    },
    defaultValue: 0,
    onChange: (newVal, oldVal) => {
        console.log(`Setting was changed from ${oldVal} to ${newVal}`);
    },
}
```

**Features**:
- **Visual slider**: Drag to set value
- **Direct input**: Type value directly
- **PrimeVue component**: Slider
- **Attributes**: `min`, `max`, `step`
- **Use cases**: Opacity, scale, intensity

#### Combo (Dropdown)
```javascript
{
    id: "example.combo",
    name: "Example combo setting",
    type: "combo",
    defaultValue: "first",
    options: [
        { text: "My first option", value: "first" },
        "My second option",
    ],
    attrs: {
        editable: true,
        filter: true,
    },
    onChange: (newVal, oldVal) => {
        console.log(`Setting was changed from ${oldVal} to ${newVal}`);
    },
}
```

**Features**:
- **Dropdown selection**: Choose from options
- **PrimeVue component**: Select
- **Options**: Objects with `text`/`value` or plain strings
- **Attributes**: `editable`, `filter`
- **Use cases**: Mode selection, preset choices

#### Color
```javascript
{
    id: "example.color",
    name: "Example color setting",
    type: "color",
    defaultValue: "ff0000",
    onChange: (newVal, oldVal) => {
        console.log(`Setting was changed from ${oldVal} to ${newVal}`);
    },
}
```

**Features**:
- **Color picker**: Visual color selection
- **Hex input**: Type hex values directly
- **PrimeVue component**: ColorPicker
- **Format**: Six hex digits (no shorthand)
- **Use cases**: Theme colors, UI customization

#### Image
```javascript
{
    id: "example.image",
    name: "Example image setting",
    type: "image",
    onChange: (newVal, oldVal) => {
        console.log(`Setting was changed from ${oldVal} to ${newVal}`);
    },
}
```

**Features**:
- **Image upload**: Upload image files
- **PrimeVue component**: FileUpload
- **Storage**: Saved as data URL
- **Use cases**: Logos, backgrounds, icons

#### Hidden
```javascript
{
    id: "example.hidden",
    name: "Example hidden setting",
    type: "hidden",
}
```

**Features**:
- **Not displayed**: Hidden from settings panel
- **Programmatic access**: Read/write from code
- **Use cases**: Internal state, cached data

### Advanced Configuration

#### Categories
```javascript
{
    id: "example.boolean",
    name: "Example boolean setting",
    type: "boolean",
    defaultValue: false,
    category: ["Category name", "Section heading", "Setting label"],
}
```

**Benefits**:
- **Custom organization**: Override ID-based categorization
- **Flexible naming**: Change without losing user values
- **Hierarchical structure**: Multiple levels of organization

#### Tooltips
```javascript
{
    id: "example.boolean",
    name: "Example boolean setting",
    type: "boolean",
    defaultValue: false,
    tooltip: "This is some helpful information",
}
```

**Features**:
- **Contextual help**: ℹ️ icon with hover text
- **User guidance**: Explain setting purpose
- **Better UX**: Reduce confusion

### PrimeVue Integration

#### Attributes
```javascript
{
    id: "example.number",
    name: "Example number setting",
    type: "number",
    defaultValue: 0,
    attrs: {
        showButtons: true,        // Show increment/decrement buttons
        maxFractionDigits: 2,     // Allow decimal places
        min: 0,                  // Minimum value
        max: 100,                // Maximum value
        step: 0.1,               // Step size
    },
}
```

**Available Attributes**:
- **Number**: `showButtons`, `maxFractionDigits`, `min`, `max`, `step`
- **Slider**: `min`, `max`, `step`
- **Combo**: `editable`, `filter`
- **Text**: `placeholder`, `maxLength`

### Complete Example

#### Extension with Settings
```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "My Extension",
    
    settings: [
        {
            id: "my.extension.enabled",
            name: "Enable Extension",
            type: "boolean",
            defaultValue: true,
            tooltip: "Enable or disable this extension",
            onChange: (newVal, oldVal) => {
                console.log(`Extension ${newVal ? 'enabled' : 'disabled'}`);
            },
        },
        {
            id: "my.extension.mode",
            name: "Operation Mode",
            type: "combo",
            defaultValue: "auto",
            options: [
                { text: "Automatic", value: "auto" },
                { text: "Manual", value: "manual" },
                { text: "Disabled", value: "disabled" },
            ],
            tooltip: "Choose how the extension operates",
            onChange: (newVal, oldVal) => {
                console.log(`Mode changed from ${oldVal} to ${newVal}`);
            },
        },
        {
            id: "my.extension.threshold",
            name: "Threshold",
            type: "slider",
            defaultValue: 0.5,
            attrs: {
                min: 0,
                max: 1,
                step: 0.01,
            },
            tooltip: "Adjust the sensitivity threshold",
            onChange: (newVal, oldVal) => {
                console.log(`Threshold changed from ${oldVal} to ${newVal}`);
            },
        },
        {
            id: "my.extension.color",
            name: "Theme Color",
            type: "color",
            defaultValue: "ff6b6b",
            tooltip: "Choose the theme color for this extension",
            onChange: (newVal, oldVal) => {
                console.log(`Color changed from ${oldVal} to ${newVal}`);
                this.updateThemeColor(newVal);
            },
        },
        {
            id: "my.extension.logo",
            name: "Logo Image",
            type: "image",
            tooltip: "Upload a logo image for this extension",
            onChange: (newVal, oldVal) => {
                console.log(`Logo changed`);
                this.updateLogo(newVal);
            },
        },
        {
            id: "my.extension.internal.state",
            name: "Internal State",
            type: "hidden",
        },
    ],
    
    async setup() {
        // Initialize extension with settings
        this.initializeWithSettings();
    },
    
    initializeWithSettings() {
        // Check if extension is enabled
        const enabled = app.extensionManager.setting.get('my.extension.enabled');
        if (!enabled) {
            console.log("Extension is disabled");
            return;
        }
        
        // Get mode setting
        const mode = app.extensionManager.setting.get('my.extension.mode');
        console.log(`Extension mode: ${mode}`);
        
        // Get threshold setting
        const threshold = app.extensionManager.setting.get('my.extension.threshold');
        console.log(`Threshold: ${threshold}`);
        
        // Get color setting
        const color = app.extensionManager.setting.get('my.extension.color');
        this.updateThemeColor(color);
        
        // Get logo setting
        const logo = app.extensionManager.setting.get('my.extension.logo');
        if (logo) {
            this.updateLogo(logo);
        }
    },
    
    updateThemeColor(color) {
        // Apply theme color
        document.documentElement.style.setProperty('--extension-color', `#${color}`);
    },
    
    updateLogo(logoDataUrl) {
        // Update logo image
        const logoElement = document.getElementById('extension-logo');
        if (logoElement) {
            logoElement.src = logoDataUrl;
        }
    },
    
    async saveInternalState(state) {
        try {
            await app.extensionManager.setting.set('my.extension.internal.state', state);
        } catch (error) {
            console.error('Error saving internal state:', error);
        }
    },
    
    getInternalState() {
        return app.extensionManager.setting.get('my.extension.internal.state');
    }
});
```

### Best Practices

#### Setting Design
```javascript
// Good: Clear, descriptive names
{
    id: "my.extension.auto.save",
    name: "Auto Save",
    type: "boolean",
    defaultValue: true,
    tooltip: "Automatically save changes when nodes are modified",
}

// Bad: Vague, unclear names
{
    id: "setting1",
    name: "Option",
    type: "boolean",
    defaultValue: false,
}
```

#### Error Handling
```javascript
async updateSetting(settingId, value) {
    try {
        await app.extensionManager.setting.set(settingId, value);
        console.log(`Setting ${settingId} updated to ${value}`);
    } catch (error) {
        console.error(`Error updating setting ${settingId}:`, error);
        // Handle error gracefully
    }
}
```

#### Performance Considerations
```javascript
// Cache frequently accessed settings
class ExtensionSettings {
    constructor() {
        this.cache = new Map();
        this.loadSettings();
    }
    
    loadSettings() {
        // Load all settings at once
        const settings = [
            'my.extension.enabled',
            'my.extension.mode',
            'my.extension.threshold',
        ];
        
        settings.forEach(settingId => {
            const value = app.extensionManager.setting.get(settingId);
            this.cache.set(settingId, value);
        });
    }
    
    get(settingId) {
        return this.cache.get(settingId);
    }
    
    async set(settingId, value) {
        try {
            await app.extensionManager.setting.set(settingId, value);
            this.cache.set(settingId, value);
        } catch (error) {
            console.error(`Error setting ${settingId}:`, error);
        }
    }
}
```

## ComfyUI Dialog API - User Interaction

### Purpose
ComfyUI provides a standardized Dialog API that works consistently across desktop and web environments. This allows extension authors to create user-friendly dialogs for input, confirmation, and other interactive elements.

### Basic Usage

#### Prompt Dialog
```javascript
// Show a prompt dialog
app.extensionManager.dialog.prompt({
  title: "User Input",
  message: "Please enter your name:",
  defaultValue: "User"
}).then(result => {
  if (result !== null) {
    console.log(`Input: ${result}`);
  }
});
```

**Features**:
- **User input**: Get text input from users
- **Default value**: Pre-fill input field
- **Cancellation**: Handle user cancellation
- **Promise-based**: Async/await compatible

#### Confirm Dialog
```javascript
// Show a confirmation dialog
app.extensionManager.dialog.confirm({
  title: "Confirm Action",
  message: "Are you sure you want to continue?",
  type: "default"
}).then(result => {
  console.log(result ? "User confirmed" : "User cancelled");
});
```

**Features**:
- **User confirmation**: Get yes/no responses
- **Multiple types**: Different dialog styles
- **Item lists**: Show lists of items
- **Hints**: Additional context information

### API Reference

#### Prompt Dialog
```javascript
app.extensionManager.dialog.prompt({
  title: string,             // Dialog title
  message: string,           // Message/question to display
  defaultValue?: string      // Initial value in the input field (optional)
}).then((result: string | null) => {
  // result is the entered text, or null if cancelled
});
```

**Parameters**:
- **title**: Dialog window title
- **message**: Question or instruction text
- **defaultValue**: Pre-filled input value (optional)
- **result**: User input string or null if cancelled

#### Confirm Dialog
```javascript
app.extensionManager.dialog.confirm({
  title: string,             // Dialog title
  message: string,           // Message to display
  type?: "default" | "overwrite" | "delete" | "dirtyClose" | "reinstall", // Dialog type (optional)
  itemList?: string[],       // List of items to display (optional)
  hint?: string              // Hint text to display (optional)
}).then((result: boolean | null) => {
  // result is true if confirmed, false if denied, null if cancelled
});
```

**Parameters**:
- **title**: Dialog window title
- **message**: Confirmation message
- **type**: Dialog style (optional)
- **itemList**: Array of items to display (optional)
- **hint**: Additional context text (optional)
- **result**: true (confirmed), false (denied), null (cancelled)

### Dialog Types

#### Default
```javascript
app.extensionManager.dialog.confirm({
  title: "Confirm Action",
  message: "Are you sure you want to continue?",
  type: "default"
}).then(result => {
  console.log(result ? "User confirmed" : "User cancelled");
});
```

**Use cases**: General confirmations, simple yes/no questions

#### Overwrite
```javascript
app.extensionManager.dialog.confirm({
  title: "File Exists",
  message: "The file already exists. Do you want to overwrite it?",
  type: "overwrite",
  itemList: ["existing_file.txt", "backup_file.txt"]
}).then(result => {
  if (result) {
    console.log("User chose to overwrite");
  }
});
```

**Use cases**: File overwrite confirmations, destructive actions

#### Delete
```javascript
app.extensionManager.dialog.confirm({
  title: "Delete Items",
  message: "Are you sure you want to delete these items?",
  type: "delete",
  itemList: ["item1.txt", "item2.txt", "item3.txt"],
  hint: "This action cannot be undone"
}).then(result => {
  if (result) {
    console.log("User confirmed deletion");
  }
});
```

**Use cases**: Deletion confirmations, permanent actions

#### Dirty Close
```javascript
app.extensionManager.dialog.confirm({
  title: "Unsaved Changes",
  message: "You have unsaved changes. Do you want to save before closing?",
  type: "dirtyClose"
}).then(result => {
  if (result) {
    console.log("User chose to save");
  }
});
```

**Use cases**: Unsaved changes, data loss prevention

#### Reinstall
```javascript
app.extensionManager.dialog.confirm({
  title: "Reinstall Extension",
  message: "Do you want to reinstall this extension?",
  type: "reinstall",
  hint: "This will remove all current settings"
}).then(result => {
  if (result) {
    console.log("User confirmed reinstall");
  }
});
```

**Use cases**: Extension management, system changes

### Advanced Usage Examples

#### Input Validation
```javascript
async function getValidatedInput() {
  let input = null;
  let isValid = false;
  
  while (!isValid) {
    input = await app.extensionManager.dialog.prompt({
      title: "Enter Email",
      message: "Please enter a valid email address:",
      defaultValue: input || ""
    });
    
    if (input === null) {
      // User cancelled
      return null;
    }
    
    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (emailRegex.test(input)) {
      isValid = true;
    } else {
      // Show error and retry
      await app.extensionManager.dialog.confirm({
        title: "Invalid Email",
        message: "Please enter a valid email address.",
        type: "default"
      });
    }
  }
  
  return input;
}
```

#### Multi-Step Confirmation
```javascript
async function confirmDestructiveAction() {
  // First confirmation
  const firstConfirm = await app.extensionManager.dialog.confirm({
    title: "Warning",
    message: "This action will permanently delete all data. Are you sure?",
    type: "delete",
    hint: "This cannot be undone"
  });
  
  if (!firstConfirm) {
    return false;
  }
  
  // Second confirmation
  const secondConfirm = await app.extensionManager.dialog.confirm({
    title: "Final Confirmation",
    message: "Are you absolutely sure? Type 'DELETE' to confirm:",
    type: "default"
  });
  
  if (!secondConfirm) {
    return false;
  }
  
  // Final text confirmation
  const textConfirm = await app.extensionManager.dialog.prompt({
    title: "Type to Confirm",
    message: "Type 'DELETE' to confirm this action:",
    defaultValue: ""
  });
  
  return textConfirm === "DELETE";
}
```

#### Batch Operations
```javascript
async function confirmBatchOperation(items) {
  const itemList = items.slice(0, 10); // Show first 10 items
  const hasMore = items.length > 10;
  
  const message = hasMore 
    ? `This will affect ${items.length} items. Continue?`
    : `This will affect the following items. Continue?`;
  
  const hint = hasMore 
    ? `... and ${items.length - 10} more items`
    : undefined;
  
  return await app.extensionManager.dialog.confirm({
    title: "Batch Operation",
    message: message,
    type: "default",
    itemList: itemList,
    hint: hint
  });
}
```

#### Settings Configuration
```javascript
async function configureExtension() {
  // Get extension name
  const name = await app.extensionManager.dialog.prompt({
    title: "Extension Configuration",
    message: "Enter extension name:",
    defaultValue: "My Extension"
  });
  
  if (name === null) return;
  
  // Get operation mode
  const mode = await app.extensionManager.dialog.confirm({
    title: "Operation Mode",
    message: "Enable automatic mode?",
    type: "default"
  });
  
  if (mode === null) return;
  
  // Get threshold value
  const threshold = await app.extensionManager.dialog.prompt({
    title: "Threshold Setting",
    message: "Enter threshold value (0.0 - 1.0):",
    defaultValue: "0.5"
  });
  
  if (threshold === null) return;
  
  // Validate threshold
  const thresholdNum = parseFloat(threshold);
  if (isNaN(thresholdNum) || thresholdNum < 0 || thresholdNum > 1) {
    await app.extensionManager.dialog.confirm({
      title: "Invalid Input",
      message: "Threshold must be a number between 0.0 and 1.0",
      type: "default"
    });
    return;
  }
  
  // Save configuration
  await app.extensionManager.setting.set('extension.name', name);
  await app.extensionManager.setting.set('extension.autoMode', mode);
  await app.extensionManager.setting.set('extension.threshold', thresholdNum);
  
  console.log("Extension configured successfully");
}
```

### Error Handling

#### Safe Dialog Usage
```javascript
async function safePrompt(title, message, defaultValue = "") {
  try {
    const result = await app.extensionManager.dialog.prompt({
      title: title,
      message: message,
      defaultValue: defaultValue
    });
    return result;
  } catch (error) {
    console.error("Dialog error:", error);
    return null;
  }
}

async function safeConfirm(title, message, type = "default") {
  try {
    const result = await app.extensionManager.dialog.confirm({
      title: title,
      message: message,
      type: type
    });
    return result;
  } catch (error) {
    console.error("Dialog error:", error);
    return null;
  }
}
```

#### Timeout Handling
```javascript
async function promptWithTimeout(title, message, timeout = 30000) {
  const promptPromise = app.extensionManager.dialog.prompt({
    title: title,
    message: message
  });
  
  const timeoutPromise = new Promise((resolve) => {
    setTimeout(() => resolve(null), timeout);
  });
  
  try {
    const result = await Promise.race([promptPromise, timeoutPromise]);
    return result;
  } catch (error) {
    console.error("Prompt timeout or error:", error);
    return null;
  }
}
```

### Best Practices

#### Dialog Design
```javascript
// Good: Clear, concise messages
app.extensionManager.dialog.confirm({
  title: "Delete File",
  message: "Are you sure you want to delete 'important_file.txt'?",
  type: "delete",
  hint: "This action cannot be undone"
});

// Bad: Vague, unclear messages
app.extensionManager.dialog.confirm({
  title: "Action",
  message: "Do you want to continue?",
  type: "default"
});
```

#### User Experience
```javascript
// Provide context and options
async function handleFileConflict(existingFile, newFile) {
  const result = await app.extensionManager.dialog.confirm({
    title: "File Conflict",
    message: `The file '${existingFile}' already exists. What would you like to do?`,
    type: "overwrite",
    itemList: [existingFile, newFile],
    hint: "Choose 'Yes' to overwrite, 'No' to skip, or 'Cancel' to abort"
  });
  
  if (result === true) {
    // Overwrite
    return "overwrite";
  } else if (result === false) {
    // Skip
    return "skip";
  } else {
    // Cancel
    return "cancel";
  }
}
```

#### Performance Considerations
```javascript
// Cache dialog results for repeated operations
class DialogCache {
  constructor() {
    this.cache = new Map();
  }
  
  async getCachedPrompt(key, title, message, defaultValue = "") {
    if (this.cache.has(key)) {
      return this.cache.get(key);
    }
    
    const result = await app.extensionManager.dialog.prompt({
      title: title,
      message: message,
      defaultValue: defaultValue
    });
    
    if (result !== null) {
      this.cache.set(key, result);
    }
    
    return result;
  }
  
  clearCache() {
    this.cache.clear();
  }
}
```

### Integration with Extensions

#### Extension Dialog Service
```javascript
app.registerExtension({
  name: "My Extension",
  
  async setup() {
    this.dialogService = new ExtensionDialogService();
  },
  
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeType.comfyClass === "MyCustomNode") {
      // Add dialog functionality to nodes
      nodeType.prototype.showConfigDialog = async function() {
        return await this.dialogService.showNodeConfigDialog(this);
      };
    }
  }
});

class ExtensionDialogService {
  async showNodeConfigDialog(node) {
    const config = await app.extensionManager.dialog.prompt({
      title: "Node Configuration",
      message: "Enter configuration for this node:",
      defaultValue: node.config || ""
    });
    
    if (config !== null) {
      node.config = config;
      node.setDirtyCanvas(true);
    }
    
    return config;
  }
  
  async confirmNodeDeletion(node) {
    return await app.extensionManager.dialog.confirm({
      title: "Delete Node",
      message: `Are you sure you want to delete '${node.title}'?`,
      type: "delete",
      hint: "This action cannot be undone"
    });
  }
}
```

## ComfyUI Toast API - User Notifications

### Purpose
ComfyUI provides a Toast API for displaying non-blocking notification messages to users. These toasts are useful for providing feedback without interrupting workflow execution.

### Basic Usage

#### Simple Toast
```javascript
// Display a simple info toast
app.extensionManager.toast.add({
  severity: "info",
  summary: "Information",
  detail: "Operation completed successfully",
  life: 3000
});
```

**Features**:
- **Non-blocking**: Doesn't interrupt user workflow
- **Auto-dismiss**: Automatically closes after specified time
- **User closable**: Users can manually close toasts
- **Multiple types**: Different severity levels

#### Toast Types
```javascript
// Success toast
app.extensionManager.toast.add({
  severity: "success",
  summary: "Success",
  detail: "Data saved successfully",
  life: 3000
});

// Warning toast
app.extensionManager.toast.add({
  severity: "warn",
  summary: "Warning",
  detail: "This action may cause problems",
  life: 5000
});

// Error toast
app.extensionManager.toast.add({
  severity: "error",
  summary: "Error",
  detail: "Failed to process request",
  life: 5000
});
```

**Severity Levels**:
- **success**: Green, for successful operations
- **info**: Blue, for informational messages
- **warn**: Orange, for warnings
- **error**: Red, for errors
- **secondary**: Gray, for secondary information
- **contrast**: High contrast, for important messages

#### Alert Helper
```javascript
// Shorthand for creating an alert toast
app.extensionManager.toast.addAlert("This is an important message");
```

**Features**:
- **Quick creation**: Simple one-line toast creation
- **Default styling**: Uses default alert styling
- **Convenient**: Perfect for simple notifications

### API Reference

#### Toast Message
```javascript
app.extensionManager.toast.add({
  severity?: "success" | "info" | "warn" | "error" | "secondary" | "contrast", // Message severity level (default: "info")
  summary?: string,         // Short title for the toast
  detail?: any,             // Detailed message content
  closable?: boolean,       // Whether user can close the toast (default: true)
  life?: number,            // Duration in milliseconds before auto-closing
  group?: string,           // Group identifier for managing related toasts
  styleClass?: any,         // Style class of the message
  contentStyleClass?: any   // Style class of the content
});
```

**Parameters**:
- **severity**: Visual style and color of the toast
- **summary**: Short title displayed prominently
- **detail**: Detailed message content
- **closable**: Whether user can manually close
- **life**: Auto-close duration in milliseconds
- **group**: Group identifier for managing related toasts
- **styleClass**: Custom CSS class for the message
- **contentStyleClass**: Custom CSS class for the content

#### Alert Helper
```javascript
app.extensionManager.toast.addAlert(message: string);
```

**Parameters**:
- **message**: The alert message to display

#### Additional Methods
```javascript
// Remove a specific toast
app.extensionManager.toast.remove(toastMessage);

// Remove all toasts
app.extensionManager.toast.removeAll();
```

### Advanced Usage Examples

#### Toast with Custom Styling
```javascript
app.extensionManager.toast.add({
  severity: "success",
  summary: "Custom Styled Toast",
  detail: "This toast has custom styling",
  life: 5000,
  styleClass: "custom-toast",
  contentStyleClass: "custom-content"
});
```

#### Grouped Toasts
```javascript
// Create toasts with the same group
app.extensionManager.toast.add({
  severity: "info",
  summary: "Processing",
  detail: "Step 1 of 3 completed",
  group: "batch-operation"
});

app.extensionManager.toast.add({
  severity: "info",
  summary: "Processing",
  detail: "Step 2 of 3 completed",
  group: "batch-operation"
});

app.extensionManager.toast.add({
  severity: "success",
  summary: "Complete",
  detail: "All steps completed successfully",
  group: "batch-operation"
});
```

#### Persistent Toasts
```javascript
// Toast that doesn't auto-close
app.extensionManager.toast.add({
  severity: "warn",
  summary: "Important Notice",
  detail: "Please review the settings before continuing",
  life: 0,  // 0 means no auto-close
  closable: true
});
```

#### Toast with Rich Content
```javascript
app.extensionManager.toast.add({
  severity: "info",
  summary: "File Upload",
  detail: `
    <div>
      <strong>File:</strong> document.pdf<br>
      <strong>Size:</strong> 2.5 MB<br>
      <strong>Status:</strong> Upload complete
    </div>
  `,
  life: 5000
});
```

### Extension Integration

#### Toast Service Class
```javascript
class ExtensionToastService {
  constructor() {
    this.toastGroup = `extension-${Date.now()}`;
  }
  
  // Success notification
  success(message, detail = null) {
    app.extensionManager.toast.add({
      severity: "success",
      summary: "Success",
      detail: detail || message,
      life: 3000,
      group: this.toastGroup
    });
  }
  
  // Error notification
  error(message, detail = null) {
    app.extensionManager.toast.add({
      severity: "error",
      summary: "Error",
      detail: detail || message,
      life: 5000,
      group: this.toastGroup
    });
  }
  
  // Warning notification
  warning(message, detail = null) {
    app.extensionManager.toast.add({
      severity: "warn",
      summary: "Warning",
      detail: detail || message,
      life: 4000,
      group: this.toastGroup
    });
  }
  
  // Info notification
  info(message, detail = null) {
    app.extensionManager.toast.add({
      severity: "info",
      summary: "Information",
      detail: detail || message,
      life: 3000,
      group: this.toastGroup
    });
  }
  
  // Progress notification
  progress(step, total, message) {
    app.extensionManager.toast.add({
      severity: "info",
      summary: "Progress",
      detail: `${message} (${step}/${total})`,
      life: 2000,
      group: this.toastGroup
    });
  }
  
  // Clear all extension toasts
  clear() {
    app.extensionManager.toast.removeAll();
  }
}
```

#### Extension with Toast Integration
```javascript
app.registerExtension({
  name: "My Extension",
  
  async setup() {
    this.toastService = new ExtensionToastService();
  },
  
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeType.comfyClass === "MyCustomNode") {
      // Add toast functionality to nodes
      nodeType.prototype.showSuccessToast = function(message) {
        this.toastService.success(message);
      };
      
      nodeType.prototype.showErrorToast = function(message) {
        this.toastService.error(message);
      };
    }
  }
});
```

### Error Handling and Feedback

#### Operation Feedback
```javascript
async function performOperation() {
  try {
    // Show start notification
    app.extensionManager.toast.add({
      severity: "info",
      summary: "Operation Started",
      detail: "Processing your request...",
      life: 2000
    });
    
    // Perform operation
    const result = await someAsyncOperation();
    
    // Show success notification
    app.extensionManager.toast.add({
      severity: "success",
      summary: "Operation Complete",
      detail: `Successfully processed ${result.count} items`,
      life: 3000
    });
    
    return result;
  } catch (error) {
    // Show error notification
    app.extensionManager.toast.add({
      severity: "error",
      summary: "Operation Failed",
      detail: `Error: ${error.message}`,
      life: 5000
    });
    
    throw error;
  }
}
```

#### Validation Feedback
```javascript
function validateInput(input) {
  if (!input) {
    app.extensionManager.toast.add({
      severity: "error",
      summary: "Validation Error",
      detail: "Input cannot be empty",
      life: 3000
    });
    return false;
  }
  
  if (input.length < 3) {
    app.extensionManager.toast.add({
      severity: "warn",
      summary: "Validation Warning",
      detail: "Input should be at least 3 characters long",
      life: 3000
    });
    return false;
  }
  
  app.extensionManager.toast.add({
    severity: "success",
    summary: "Validation Passed",
    detail: "Input is valid",
    life: 2000
  });
  
  return true;
}
```

#### Batch Operation Feedback
```javascript
async function processBatch(items) {
  const total = items.length;
  let processed = 0;
  let errors = 0;
  
  for (const item of items) {
    try {
      await processItem(item);
      processed++;
      
      // Show progress
      app.extensionManager.toast.add({
        severity: "info",
        summary: "Processing",
        detail: `Processed ${processed}/${total} items`,
        life: 1000,
        group: "batch-operation"
      });
    } catch (error) {
      errors++;
      console.error(`Error processing item ${item}:`, error);
    }
  }
  
  // Show final result
  if (errors === 0) {
    app.extensionManager.toast.add({
      severity: "success",
      summary: "Batch Complete",
      detail: `Successfully processed all ${total} items`,
      life: 3000,
      group: "batch-operation"
    });
  } else {
    app.extensionManager.toast.add({
      severity: "warn",
      summary: "Batch Complete with Errors",
      detail: `Processed ${processed}/${total} items, ${errors} errors`,
      life: 5000,
      group: "batch-operation"
    });
  }
}
```

### Best Practices

#### Toast Design
```javascript
// Good: Clear, actionable messages
app.extensionManager.toast.add({
  severity: "success",
  summary: "File Saved",
  detail: "document.pdf has been saved to the Documents folder",
  life: 3000
});

// Bad: Vague, unhelpful messages
app.extensionManager.toast.add({
  severity: "info",
  summary: "Done",
  detail: "OK",
  life: 3000
});
```

#### User Experience
```javascript
// Provide context and next steps
app.extensionManager.toast.add({
  severity: "warn",
  summary: "Settings Changed",
  detail: "Please restart the application for changes to take effect",
  life: 5000
});

// Use appropriate severity levels
app.extensionManager.toast.add({
  severity: "error",
  summary: "Critical Error",
  detail: "Unable to connect to server. Please check your internet connection.",
  life: 0  // Don't auto-close critical errors
});
```

#### Performance Considerations
```javascript
// Avoid toast spam
class ToastThrottler {
  constructor() {
    this.lastToast = new Map();
    this.throttleTime = 1000; // 1 second
  }
  
  addToast(options) {
    const key = `${options.severity}-${options.summary}`;
    const now = Date.now();
    const lastTime = this.lastToast.get(key) || 0;
    
    if (now - lastTime < this.throttleTime) {
      return; // Skip duplicate toast
    }
    
    this.lastToast.set(key, now);
    app.extensionManager.toast.add(options);
  }
}
```

#### Toast Management
```javascript
// Manage toast lifecycle
class ToastManager {
  constructor() {
    this.activeToasts = new Set();
  }
  
  addToast(options) {
    const toast = app.extensionManager.toast.add(options);
    this.activeToasts.add(toast);
    
    // Auto-remove from tracking when toast expires
    if (options.life > 0) {
      setTimeout(() => {
        this.activeToasts.delete(toast);
      }, options.life);
    }
    
    return toast;
  }
  
  clearAll() {
    app.extensionManager.toast.removeAll();
    this.activeToasts.clear();
  }
  
  getActiveCount() {
    return this.activeToasts.size;
  }
}
```

## ComfyUI About Panel Badges - Extension Branding

### Purpose
ComfyUI's About Panel Badges API allows extensions to add custom badges to the ComfyUI about page. These badges can display information about your extension and contain links to documentation, source code, or other resources.

### Basic Usage

#### Simple Badge Registration
```javascript
app.registerExtension({
  name: "MyExtension",
  aboutPageBadges: [
    {
      label: "Documentation",
      url: "https://example.com/docs",
      icon: "pi pi-file"
    },
    {
      label: "GitHub",
      url: "https://github.com/username/repo",
      icon: "pi pi-github"
    }
  ]
});
```

**Features**:
- **Custom branding**: Add your extension's links to the about page
- **Professional appearance**: Integrated with ComfyUI's about panel
- **User accessibility**: Easy access to extension resources
- **Icon support**: Visual indicators for different link types

### Badge Configuration

#### Required Properties
```javascript
{
  label: string,           // Text to display on the badge
  url: string,             // URL to open when badge is clicked
  icon: string             // Icon class (e.g., PrimeVue icon)
}
```

**Properties**:
- **label**: Display text for the badge
- **url**: Link destination when clicked
- **icon**: PrimeVue icon class for visual representation

### Icon Options

#### Commonly Used Icons
```javascript
// Documentation icons
"pi pi-file"        // File icon
"pi pi-book"        // Book icon

// Platform icons
"pi pi-github"      // GitHub icon
"pi pi-discord"     // Discord icon
"pi pi-globe"       // Website icon

// Action icons
"pi pi-external-link"  // External link icon
"pi pi-download"       // Download icon
"pi pi-heart"          // Donate/Heart icon
"pi pi-info-circle"    // Information icon
"pi pi-home"           // Home icon
```

**Icon Sources**:
- **PrimeVue Icons**: Complete icon set available
- **Consistent styling**: Matches ComfyUI's design system
- **Scalable**: Vector-based icons for all screen sizes

### Complete Example

#### Extension with Multiple Badges
```javascript
app.registerExtension({
  name: "BadgeExample",
  aboutPageBadges: [
    {
      label: "Website",
      url: "https://example.com",
      icon: "pi pi-home"
    },
    {
      label: "Donate",
      url: "https://example.com/donate",
      icon: "pi pi-heart"
    },
    {
      label: "Documentation",
      url: "https://example.com/docs",
      icon: "pi pi-book"
    },
    {
      label: "GitHub",
      url: "https://github.com/username/repo",
      icon: "pi pi-github"
    },
    {
      label: "Discord",
      url: "https://discord.gg/example",
      icon: "pi pi-discord"
    }
  ]
});
```

### Best Practices

#### Badge Design
```javascript
// Good: Clear, descriptive labels
{
  label: "Documentation",
  url: "https://example.com/docs",
  icon: "pi pi-book"
}

// Bad: Vague, unclear labels
{
  label: "Link",
  url: "https://example.com",
  icon: "pi pi-external-link"
}
```

#### URL Management
```javascript
// Use HTTPS for security
{
  label: "GitHub",
  url: "https://github.com/username/repo",
  icon: "pi pi-github"
}

// Avoid HTTP for security
{
  label: "Website",
  url: "http://example.com",  // Bad: HTTP
  icon: "pi pi-globe"
}
```

#### Icon Selection
```javascript
// Choose appropriate icons
{
  label: "Documentation",
  url: "https://example.com/docs",
  icon: "pi pi-book"  // Good: Book for documentation
}

{
  label: "Documentation",
  url: "https://example.com/docs",
  icon: "pi pi-heart"  // Bad: Heart doesn't represent docs
}
```

## ComfyUI Bottom Panel Tabs - Custom Panels

### Purpose
ComfyUI's Bottom Panel Tabs API allows extensions to add custom tabs to the bottom panel of the ComfyUI interface. This is useful for adding features like logs, debugging tools, or custom panels.

### Basic Usage

#### Simple Tab Registration
```javascript
app.registerExtension({
  name: "MyExtension",
  bottomPanelTabs: [
    {
      id: "customTab",
      title: "Custom Tab",
      type: "custom",
      render: (el) => {
        el.innerHTML = '<div>This is my custom tab content</div>';
      }
    }
  ]
});
```

**Features**:
- **Custom panels**: Add your own interface elements
- **Integrated UI**: Seamlessly integrated with ComfyUI
- **Flexible content**: HTML, React, or any DOM content
- **Event handling**: Full JavaScript event support

### Tab Configuration

#### Required Properties
```javascript
{
  id: string,              // Unique identifier for the tab
  title: string,           // Display title shown on the tab
  type: string,            // Tab type (usually "custom")
  icon?: string,           // Icon class (optional)
  render: (element) => void // Function that populates the tab content
}
```

**Properties**:
- **id**: Unique identifier for the tab
- **title**: Display title shown on the tab
- **type**: Tab type (usually "custom")
- **icon**: Optional icon class
- **render**: Function that populates the tab content

### Interactive Elements

#### Buttons and Controls
```javascript
app.registerExtension({
  name: "InteractiveTabExample",
  bottomPanelTabs: [
    {
      id: "controlsTab",
      title: "Controls",
      type: "custom",
      render: (el) => {
        el.innerHTML = `
          <div style="padding: 10px;">
            <button id="runBtn">Run Workflow</button>
            <button id="clearBtn">Clear Queue</button>
            <input type="text" id="textInput" placeholder="Enter text...">
          </div>
        `;
        
        // Add event listeners
        el.querySelector('#runBtn').addEventListener('click', () => {
          app.queuePrompt();
        });
        
        el.querySelector('#clearBtn').addEventListener('click', () => {
          app.clearQueue();
        });
        
        el.querySelector('#textInput').addEventListener('input', (e) => {
          console.log('Text input:', e.target.value);
        });
      }
    }
  ]
});
```

### Using React Components

#### React Integration
```javascript
// Import React dependencies in your extension
import React from "react";
import ReactDOM from "react-dom/client";

// Simple React component
function TabContent() {
  const [count, setCount] = React.useState(0);
  const [text, setText] = React.useState("");
  
  return (
    <div style={{ padding: "10px" }}>
      <h3>React Component</h3>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <br />
      <input 
        type="text" 
        value={text} 
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text..."
      />
      <p>You typed: {text}</p>
    </div>
  );
}

// Register the extension with React content
app.registerExtension({
  name: "ReactTabExample",
  bottomPanelTabs: [
    {
      id: "reactTab",
      title: "React Tab",
      type: "custom",
      render: (el) => {
        const container = document.createElement("div");
        container.id = "react-tab-container";
        el.appendChild(container);
        
        // Mount React component
        ReactDOM.createRoot(container).render(
          <React.StrictMode>
            <TabContent />
          </React.StrictMode>
        );
      }
    }
  ]
});
```

### Standalone Registration

#### Independent Tab Registration
```javascript
app.extensionManager.registerBottomPanelTab({
  id: "standAloneTab",
  title: "Stand-Alone Tab",
  type: "custom",
  render: (el) => {
    el.innerHTML = '<div>This tab was registered independently</div>';
  }
});
```

**Benefits**:
- **Flexible registration**: Register tabs outside of extensions
- **Dynamic creation**: Create tabs at runtime
- **Modular approach**: Separate tab logic from extension logic

### Advanced Examples

#### Log Viewer Tab
```javascript
app.registerExtension({
  name: "LogViewer",
  bottomPanelTabs: [
    {
      id: "logViewer",
      title: "Log Viewer",
      type: "custom",
      render: (el) => {
        el.innerHTML = `
          <div style="padding: 10px;">
            <h3>Application Logs</h3>
            <div id="logContainer" style="height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; background: #f5f5f5;">
              <div>Log entries will appear here...</div>
            </div>
            <button id="clearLogs">Clear Logs</button>
            <button id="exportLogs">Export Logs</button>
          </div>
        `;
        
        const logContainer = el.querySelector('#logContainer');
        const clearBtn = el.querySelector('#clearLogs');
        const exportBtn = el.querySelector('#exportLogs');
        
        // Add log entry
        function addLogEntry(message, type = 'info') {
          const entry = document.createElement('div');
          entry.style.marginBottom = '5px';
          entry.style.padding = '5px';
          entry.style.borderRadius = '3px';
          entry.style.backgroundColor = type === 'error' ? '#ffebee' : '#e8f5e8';
          entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
          logContainer.appendChild(entry);
          logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        // Clear logs
        clearBtn.addEventListener('click', () => {
          logContainer.innerHTML = '<div>Logs cleared</div>';
        });
        
        // Export logs
        exportBtn.addEventListener('click', () => {
          const logs = logContainer.textContent;
          const blob = new Blob([logs], { type: 'text/plain' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'comfyui-logs.txt';
          a.click();
          URL.revokeObjectURL(url);
        });
        
        // Simulate log entries
        setInterval(() => {
          addLogEntry('System check completed', 'info');
        }, 5000);
      }
    }
  ]
});
```

#### Settings Panel Tab
```javascript
app.registerExtension({
  name: "SettingsPanel",
  bottomPanelTabs: [
    {
      id: "settingsPanel",
      title: "Settings",
      type: "custom",
      render: (el) => {
        el.innerHTML = `
          <div style="padding: 10px;">
            <h3>Extension Settings</h3>
            <div style="margin-bottom: 10px;">
              <label>
                <input type="checkbox" id="autoSave"> Auto Save
              </label>
            </div>
            <div style="margin-bottom: 10px;">
              <label>
                Theme: 
                <select id="themeSelect">
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                  <option value="auto">Auto</option>
                </select>
              </label>
            </div>
            <div style="margin-bottom: 10px;">
              <label>
                Refresh Rate: 
                <input type="range" id="refreshRate" min="1" max="10" value="5">
                <span id="refreshValue">5</span> seconds
              </label>
            </div>
            <button id="saveSettings">Save Settings</button>
            <button id="resetSettings">Reset to Defaults</button>
          </div>
        `;
        
        // Load saved settings
        const autoSave = el.querySelector('#autoSave');
        const themeSelect = el.querySelector('#themeSelect');
        const refreshRate = el.querySelector('#refreshRate');
        const refreshValue = el.querySelector('#refreshValue');
        const saveBtn = el.querySelector('#saveSettings');
        const resetBtn = el.querySelector('#resetSettings');
        
        // Load settings
        autoSave.checked = localStorage.getItem('autoSave') === 'true';
        themeSelect.value = localStorage.getItem('theme') || 'auto';
        refreshRate.value = localStorage.getItem('refreshRate') || '5';
        refreshValue.textContent = refreshRate.value;
        
        // Update refresh rate display
        refreshRate.addEventListener('input', (e) => {
          refreshValue.textContent = e.target.value;
        });
        
        // Save settings
        saveBtn.addEventListener('click', () => {
          localStorage.setItem('autoSave', autoSave.checked);
          localStorage.setItem('theme', themeSelect.value);
          localStorage.setItem('refreshRate', refreshRate.value);
          alert('Settings saved!');
        });
        
        // Reset settings
        resetBtn.addEventListener('click', () => {
          autoSave.checked = false;
          themeSelect.value = 'auto';
          refreshRate.value = '5';
          refreshValue.textContent = '5';
          localStorage.clear();
          alert('Settings reset to defaults!');
        });
      }
    }
  ]
});
```

### Best Practices

#### Tab Design
```javascript
// Good: Clear, descriptive titles
{
  id: "logViewer",
  title: "Log Viewer",
  type: "custom",
  render: (el) => { /* ... */ }
}

// Bad: Vague, unclear titles
{
  id: "tab1",
  title: "Tab",
  type: "custom",
  render: (el) => { /* ... */ }
}
```

#### Content Organization
```javascript
// Organize content with proper styling
render: (el) => {
  el.innerHTML = `
    <div style="padding: 10px;">
      <h3>Panel Title</h3>
      <div style="margin-bottom: 10px;">
        <label>Setting 1:</label>
        <input type="text" id="setting1">
      </div>
      <div style="margin-bottom: 10px;">
        <label>Setting 2:</label>
        <select id="setting2">
          <option value="option1">Option 1</option>
          <option value="option2">Option 2</option>
        </select>
      </div>
      <button id="saveBtn">Save</button>
    </div>
  `;
}
```

#### Event Handling
```javascript
// Clean up event listeners
render: (el) => {
  // Store references for cleanup
  const button = document.createElement('button');
  button.textContent = 'Click Me';
  
  const handleClick = () => {
    console.log('Button clicked');
  };
  
  button.addEventListener('click', handleClick);
  el.appendChild(button);
  
  // Cleanup function (if needed)
  el.cleanup = () => {
    button.removeEventListener('click', handleClick);
  };
}
```

## ComfyUI Sidebar Tabs - Custom Sidebar Panels

### Purpose
ComfyUI's Sidebar Tabs API allows extensions to add custom tabs to the sidebar of the ComfyUI interface. This is useful for adding features that require persistent visibility and quick access.

### Basic Usage

#### Simple Sidebar Tab Registration
```javascript
app.extensionManager.registerSidebarTab({
  id: "customSidebar",
  icon: "pi pi-compass",
  title: "Custom Tab",
  tooltip: "My Custom Sidebar Tab",
  type: "custom",
  render: (el) => {
    el.innerHTML = '<div>This is my custom sidebar content</div>';
  }
});
```

**Features**:
- **Persistent visibility**: Always accessible in the sidebar
- **Quick access**: Easy to reach from anywhere in the interface
- **Custom content**: Full control over tab content
- **Icon support**: Visual indicators for different tabs

### Tab Configuration

#### Required Properties
```javascript
{
  id: string,              // Unique identifier for the tab
  icon: string,            // Icon class for the tab button
  title: string,           // Title text for the tab
  tooltip?: string,        // Tooltip text on hover (optional)
  type: string,            // Tab type (usually "custom")
  render: (element) => void // Function that populates the tab content
}
```

**Properties**:
- **id**: Unique identifier for the tab
- **icon**: Icon class for the tab button
- **title**: Title text for the tab
- **tooltip**: Optional tooltip text on hover
- **type**: Tab type (usually "custom")
- **render**: Function that populates the tab content

### Icon Options

#### Icon Libraries
```javascript
// PrimeVue icons
"pi pi-home"           // Home icon
"pi pi-compass"        // Compass icon
"pi pi-list"           // List icon
"pi pi-chart-line"     // Chart icon
"pi pi-cog"            // Settings icon

// Material Design icons
"mdi mdi-robot"        // Robot icon
"mdi mdi-react"        // React icon
"mdi mdi-github"        // GitHub icon
"mdi mdi-database"      // Database icon

// Font Awesome icons
"fa-solid fa-star"     // Star icon
"fa-regular fa-heart"   // Heart icon
"fa-brands fa-github"   // GitHub icon
```

**Icon Requirements**:
- **Library loading**: Ensure corresponding icon library is loaded
- **Consistent styling**: Use icons that match ComfyUI's design
- **Appropriate selection**: Choose icons that represent tab purpose

### Stateful Tab Example

#### Notes Tab with Persistence
```javascript
app.extensionManager.registerSidebarTab({
  id: "statefulTab",
  icon: "pi pi-list",
  title: "Notes",
  type: "custom",
  render: (el) => {
    // Create elements
    const container = document.createElement('div');
    container.style.padding = '10px';
    
    const notepad = document.createElement('textarea');
    notepad.style.width = '100%';
    notepad.style.height = '200px';
    notepad.style.marginBottom = '10px';
    notepad.placeholder = 'Enter your notes here...';
    
    const saveBtn = document.createElement('button');
    saveBtn.textContent = 'Save Notes';
    saveBtn.style.marginRight = '10px';
    
    const clearBtn = document.createElement('button');
    clearBtn.textContent = 'Clear Notes';
    
    // Load saved content if available
    const savedContent = localStorage.getItem('comfyui-notes');
    if (savedContent) {
      notepad.value = savedContent;
    }
    
    // Auto-save content
    notepad.addEventListener('input', () => {
      localStorage.setItem('comfyui-notes', notepad.value);
    });
    
    // Manual save
    saveBtn.addEventListener('click', () => {
      localStorage.setItem('comfyui-notes', notepad.value);
      alert('Notes saved!');
    });
    
    // Clear notes
    clearBtn.addEventListener('click', () => {
      if (confirm('Are you sure you want to clear all notes?')) {
        notepad.value = '';
        localStorage.removeItem('comfyui-notes');
      }
    });
    
    // Assemble the UI
    container.appendChild(notepad);
    container.appendChild(saveBtn);
    container.appendChild(clearBtn);
    el.appendChild(container);
  }
});
```

### Using React Components

#### React Integration
```javascript
// Import React dependencies in your extension
import React from "react";
import ReactDOM from "react-dom/client";

// Register sidebar tab with React content
app.extensionManager.registerSidebarTab({
  id: "reactSidebar",
  icon: "mdi mdi-react",
  title: "React Tab",
  type: "custom",
  render: (el) => {
    const container = document.createElement("div");
    container.id = "react-sidebar-container";
    el.appendChild(container);
    
    // Define a simple React component
    function SidebarContent() {
      const [count, setCount] = React.useState(0);
      const [text, setText] = React.useState("");
      
      return (
        <div style={{ padding: "10px" }}>
          <h3>React Sidebar</h3>
          <p>Count: {count}</p>
          <button onClick={() => setCount(count + 1)}>
            Increment
          </button>
          <br />
          <input 
            type="text" 
            value={text} 
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text..."
            style={{ marginTop: "10px", width: "100%" }}
          />
          <p>You typed: {text}</p>
        </div>
      );
    }
    
    // Mount React component
    ReactDOM.createRoot(container).render(
      <React.StrictMode>
        <SidebarContent />
      </React.StrictMode>
    );
  }
});
```

### Dynamic Content Updates

#### Stats Tab with Real-time Updates
```javascript
app.extensionManager.registerSidebarTab({
  id: "dynamicSidebar",
  icon: "pi pi-chart-line",
  title: "Stats",
  type: "custom",
  render: (el) => {
    const container = document.createElement('div');
    container.style.padding = '10px';
    el.appendChild(container);
    
    // Function to update stats
    function updateStats() {
      const stats = {
        nodes: app.graph._nodes.length,
        connections: Object.keys(app.graph.links).length,
        groups: app.graph._groups ? app.graph._groups.length : 0
      };
      
      container.innerHTML = `
        <h3>Workflow Stats</h3>
        <ul>
          <li>Nodes: ${stats.nodes}</li>
          <li>Connections: ${stats.connections}</li>
          <li>Groups: ${stats.groups}</li>
        </ul>
        <button id="refreshStats">Refresh</button>
      `;
      
      // Add refresh button functionality
      const refreshBtn = container.querySelector('#refreshStats');
      if (refreshBtn) {
        refreshBtn.addEventListener('click', updateStats);
      }
    }
    
    // Initial update
    updateStats();
    
    // Listen for graph changes
    const api = app.api;
    api.addEventListener("graphChanged", updateStats);
    
    // Clean up listeners when tab is destroyed
    return () => {
      api.removeEventListener("graphChanged", updateStats);
    };
  }
});
```

### Advanced Examples

#### File Manager Sidebar
```javascript
app.extensionManager.registerSidebarTab({
  id: "fileManager",
  icon: "pi pi-folder",
  title: "Files",
  type: "custom",
  render: (el) => {
    const container = document.createElement('div');
    container.style.padding = '10px';
    el.appendChild(container);
    
    // File list container
    const fileList = document.createElement('div');
    fileList.style.height = '300px';
    fileList.style.overflowY = 'auto';
    fileList.style.border = '1px solid #ccc';
    fileList.style.padding = '10px';
    fileList.style.marginBottom = '10px';
    
    // Controls
    const controls = document.createElement('div');
    controls.style.marginBottom = '10px';
    
    const uploadBtn = document.createElement('button');
    uploadBtn.textContent = 'Upload File';
    uploadBtn.style.marginRight = '10px';
    
    const refreshBtn = document.createElement('button');
    refreshBtn.textContent = 'Refresh';
    
    // File input (hidden)
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.style.display = 'none';
    fileInput.multiple = true;
    
    // Load files
    function loadFiles() {
      // Simulate file loading
      const files = [
        { name: 'image1.jpg', size: '2.5 MB', type: 'image' },
        { name: 'document.pdf', size: '1.2 MB', type: 'document' },
        { name: 'data.json', size: '0.5 MB', type: 'data' }
      ];
      
      fileList.innerHTML = files.map(file => `
        <div style="padding: 5px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between;">
          <span>${file.name}</span>
          <span style="color: #666;">${file.size}</span>
        </div>
      `).join('');
    }
    
    // Upload files
    uploadBtn.addEventListener('click', () => {
      fileInput.click();
    });
    
    fileInput.addEventListener('change', (e) => {
      const files = Array.from(e.target.files);
      files.forEach(file => {
        console.log('Uploaded file:', file.name);
      });
      loadFiles();
    });
    
    // Refresh files
    refreshBtn.addEventListener('click', loadFiles);
    
    // Assemble UI
    controls.appendChild(uploadBtn);
    controls.appendChild(refreshBtn);
    container.appendChild(controls);
    container.appendChild(fileList);
    container.appendChild(fileInput);
    
    // Load initial files
    loadFiles();
  }
});
```

#### Debug Console Sidebar
```javascript
app.extensionManager.registerSidebarTab({
  id: "debugConsole",
  icon: "pi pi-bug",
  title: "Debug",
  type: "custom",
  render: (el) => {
    const container = document.createElement('div');
    container.style.padding = '10px';
    el.appendChild(container);
    
    // Console output
    const consoleOutput = document.createElement('div');
    consoleOutput.style.height = '300px';
    consoleOutput.style.overflowY = 'auto';
    consoleOutput.style.border = '1px solid #ccc';
    consoleOutput.style.padding = '10px';
    consoleOutput.style.backgroundColor = '#1e1e1e';
    consoleOutput.style.color = '#ffffff';
    consoleOutput.style.fontFamily = 'monospace';
    consoleOutput.style.fontSize = '12px';
    
    // Controls
    const controls = document.createElement('div');
    controls.style.marginBottom = '10px';
    
    const clearBtn = document.createElement('button');
    clearBtn.textContent = 'Clear';
    clearBtn.style.marginRight = '10px';
    
    const logBtn = document.createElement('button');
    logBtn.textContent = 'Test Log';
    logBtn.style.marginRight = '10px';
    
    const errorBtn = document.createElement('button');
    errorBtn.textContent = 'Test Error';
    
    // Log function
    function log(message, type = 'log') {
      const timestamp = new Date().toLocaleTimeString();
      const logEntry = document.createElement('div');
      logEntry.style.marginBottom = '2px';
      logEntry.style.padding = '2px';
      
      const color = type === 'error' ? '#ff6b6b' : 
                   type === 'warn' ? '#ffa726' : '#4caf50';
      
      logEntry.innerHTML = `
        <span style="color: #666;">[${timestamp}]</span>
        <span style="color: ${color};">[${type.toUpperCase()}]</span>
        <span>${message}</span>
      `;
      
      consoleOutput.appendChild(logEntry);
      consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }
    
    // Clear console
    clearBtn.addEventListener('click', () => {
      consoleOutput.innerHTML = '';
    });
    
    // Test log
    logBtn.addEventListener('click', () => {
      log('This is a test log message');
    });
    
    // Test error
    errorBtn.addEventListener('click', () => {
      log('This is a test error message', 'error');
    });
    
    // Assemble UI
    controls.appendChild(clearBtn);
    controls.appendChild(logBtn);
    controls.appendChild(errorBtn);
    container.appendChild(controls);
    container.appendChild(consoleOutput);
    
    // Initial log
    log('Debug console initialized');
  }
});
```

### Best Practices

#### Tab Design
```javascript
// Good: Clear, descriptive titles and icons
app.extensionManager.registerSidebarTab({
  id: "fileManager",
  icon: "pi pi-folder",
  title: "File Manager",
  tooltip: "Manage your files and assets",
  type: "custom",
  render: (el) => { /* ... */ }
});

// Bad: Vague, unclear titles
app.extensionManager.registerSidebarTab({
  id: "tab1",
  icon: "pi pi-circle",
  title: "Tab",
  type: "custom",
  render: (el) => { /* ... */ }
});
```

#### Content Organization
```javascript
// Organize content with proper styling and structure
render: (el) => {
  const container = document.createElement('div');
  container.style.padding = '10px';
  
  // Header
  const header = document.createElement('h3');
  header.textContent = 'Panel Title';
  header.style.marginTop = '0';
  header.style.marginBottom = '15px';
  
  // Content area
  const content = document.createElement('div');
  content.style.marginBottom = '10px';
  
  // Controls
  const controls = document.createElement('div');
  controls.style.marginTop = '10px';
  
  // Assemble
  container.appendChild(header);
  container.appendChild(content);
  container.appendChild(controls);
  el.appendChild(container);
}
```

#### Event Handling and Cleanup
```javascript
// Proper event handling and cleanup
render: (el) => {
  const container = document.createElement('div');
  el.appendChild(container);
  
  // Store event listeners for cleanup
  const eventListeners = [];
  
  const button = document.createElement('button');
  button.textContent = 'Click Me';
  
  const handleClick = () => {
    console.log('Button clicked');
  };
  
  button.addEventListener('click', handleClick);
  eventListeners.push({ element: button, event: 'click', handler: handleClick });
  
  container.appendChild(button);
  
  // Return cleanup function
  return () => {
    eventListeners.forEach(({ element, event, handler }) => {
      element.removeEventListener(event, handler);
    });
  };
}
```

#### Performance Considerations
```javascript
// Efficient content updates
function createEfficientTab() {
  let updateTimeout;
  
  return {
    id: "efficientTab",
    icon: "pi pi-chart-line",
    title: "Efficient Tab",
    type: "custom",
    render: (el) => {
      const container = document.createElement('div');
      el.appendChild(container);
      
      // Debounced update function
      function updateContent() {
        clearTimeout(updateTimeout);
        updateTimeout = setTimeout(() => {
          // Perform expensive operations here
          container.innerHTML = `<div>Updated: ${new Date().toLocaleTimeString()}</div>`;
        }, 100);
      }
      
      // Listen for changes
      app.api.addEventListener("graphChanged", updateContent);
      
      // Initial update
      updateContent();
      
      // Cleanup
      return () => {
        clearTimeout(updateTimeout);
        app.api.removeEventListener("graphChanged", updateContent);
      };
    }
  };
}
```

## ComfyUI Selection Toolbox - Context-Sensitive Actions

### Purpose
ComfyUI's Selection Toolbox API allows extensions to add custom action buttons that appear when nodes are selected on the canvas. This provides quick access to context-sensitive commands for selected items (nodes, groups, etc.).

### Basic Usage

#### Simple Selection Toolbox Registration
```javascript
app.registerExtension({
  name: "MyExtension",
  commands: [
    {
      id: "my-extension.duplicate-special",
      label: "Duplicate Special",
      icon: "pi pi-copy",
      function: (selectedItem) => {
        // Your command logic here
        console.log("Duplicating selected nodes with special behavior");
      }
    }
  ],
  getSelectionToolboxCommands: (selectedItem) => {
    // Return array of command IDs to show in the toolbox
    return ["my-extension.duplicate-special"];
  }
});
```

**Features**:
- **Context-sensitive**: Commands appear based on selection
- **Quick access**: Immediate action buttons for selected items
- **Dynamic visibility**: Commands can change based on selection
- **Standard interface**: Uses ComfyUI's command system

### Command Definition

#### Required Properties
```javascript
{
  id: string,          // Unique identifier for the command
  label: string,       // Display text for the button tooltip
  icon?: string,       // Icon class for the button (optional)
  function: (selectedItem) => void  // Function executed when clicked
}
```

**Properties**:
- **id**: Unique identifier for the command
- **label**: Display text for the button tooltip
- **icon**: Optional icon class for the button
- **function**: Function executed when clicked, receives selected item(s)

### Icon Options

#### Supported Icon Libraries
```javascript
// PrimeVue icons
"pi pi-copy"           // Copy icon
"pi pi-align-left"     // Align left icon
"pi pi-cog"            // Settings icon
"pi pi-info-circle"    // Information icon
"pi pi-hashtag"        // Number icon
"pi pi-star"           // Star icon

// Material Design icons
"mdi mdi-content-copy"  // Copy icon
"mdi mdi-align-horizontal-left"  // Align icon
"mdi mdi-cog"          // Settings icon
"mdi mdi-information"  // Information icon
```

### Dynamic Command Visibility

#### Context-Sensitive Commands
```javascript
app.registerExtension({
  name: "ContextualCommands",
  commands: [
    {
      id: "my-ext.align-nodes",
      label: "Align Nodes",
      icon: "pi pi-align-left",
      function: () => {
        // Align multiple nodes
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        console.log(`Aligning ${selectedItems.length} nodes`);
      }
    },
    {
      id: "my-ext.configure-single",
      label: "Configure",
      icon: "pi pi-cog",
      function: () => {
        // Configure single node
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        if (selectedItems.length === 1) {
          console.log(`Configuring node: ${selectedItems[0].type}`);
        }
      }
    },
    {
      id: "my-ext.group-selection",
      label: "Group Selection",
      icon: "pi pi-objects-column",
      function: () => {
        // Group multiple items
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        console.log(`Grouping ${selectedItems.length} items`);
      }
    }
  ],
  getSelectionToolboxCommands: (selectedItem) => {
    const selectedItems = app.canvas.selectedItems;
    const itemCount = selectedItems ? selectedItems.size : 0;
    
    if (itemCount > 1) {
      // Show alignment and grouping commands for multiple items
      return ["my-ext.align-nodes", "my-ext.group-selection"];
    } else if (itemCount === 1) {
      // Show configuration for single item
      return ["my-ext.configure-single"];
    }
    
    return [];
  }
});
```

### Working with Selected Items

#### Selection Information Access
```javascript
app.registerExtension({
  name: "SelectionInfo",
  commands: [
    {
      id: "my-ext.show-info",
      label: "Show Selection Info",
      icon: "pi pi-info-circle",
      function: () => {
        const selectedItems = app.canvas.selectedItems;
        
        if (selectedItems && selectedItems.size > 0) {
          console.log(`Selected ${selectedItems.size} items`);
          
          // Iterate through selected items
          selectedItems.forEach(item => {
            if (item.type) {
              console.log(`Item: ${item.type} (ID: ${item.id})`);
            }
          });
        }
      }
    },
    {
      id: "my-ext.get-positions",
      label: "Get Positions",
      icon: "pi pi-map-marker",
      function: () => {
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        const positions = selectedItems.map(item => ({
          id: item.id,
          type: item.type,
          pos: item.pos
        }));
        
        console.log("Selected item positions:", positions);
      }
    }
  ],
  getSelectionToolboxCommands: () => ["my-ext.show-info", "my-ext.get-positions"]
});
```

### Complete Example

#### Selection Tools Extension
```javascript
app.registerExtension({
  name: "SelectionTools",
  commands: [
    {
      id: "selection-tools.count",
      label: "Count Selection",
      icon: "pi pi-hashtag",
      function: () => {
        const count = app.canvas.selectedItems?.size || 0;
        app.extensionManager.toast.add({
          severity: "info",
          summary: "Selection Count",
          detail: `You have ${count} item${count !== 1 ? 's' : ''} selected`,
          life: 3000
        });
      }
    },
    {
      id: "selection-tools.copy-ids",
      label: "Copy IDs",
      icon: "pi pi-copy",
      function: () => {
        const items = Array.from(app.canvas.selectedItems || []);
        const ids = items.map(item => item.id).filter(id => id !== undefined);
        
        if (ids.length > 0) {
          navigator.clipboard.writeText(ids.join(', '));
          app.extensionManager.toast.add({
            severity: "success",
            summary: "Copied",
            detail: `Copied ${ids.length} IDs to clipboard`,
            life: 2000
          });
        }
      }
    },
    {
      id: "selection-tools.log-types",
      label: "Log Types",
      icon: "pi pi-info-circle",
      function: () => {
        const items = Array.from(app.canvas.selectedItems || []);
        const typeCount = {};
        
        items.forEach(item => {
          const type = item.type || 'unknown';
          typeCount[type] = (typeCount[type] || 0) + 1;
        });
        
        console.log("Selection types:", typeCount);
      }
    },
    {
      id: "selection-tools.duplicate",
      label: "Duplicate Selection",
      icon: "pi pi-clone",
      function: () => {
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        console.log(`Duplicating ${selectedItems.length} items`);
        
        // Implement duplication logic here
        selectedItems.forEach(item => {
          console.log(`Duplicating: ${item.type} (ID: ${item.id})`);
        });
      }
    },
    {
      id: "selection-tools.delete",
      label: "Delete Selection",
      icon: "pi pi-trash",
      function: () => {
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        const count = selectedItems.length;
        
        if (confirm(`Are you sure you want to delete ${count} item${count !== 1 ? 's' : ''}?`)) {
          console.log(`Deleting ${count} items`);
          // Implement deletion logic here
        }
      }
    }
  ],
  
  getSelectionToolboxCommands: (selectedItem) => {
    const selectedItems = app.canvas.selectedItems;
    const itemCount = selectedItems ? selectedItems.size : 0;
    
    if (itemCount === 0) return [];
    
    const commands = ["selection-tools.count", "selection-tools.log-types"];
    
    // Only show copy command if items have IDs
    const hasIds = Array.from(selectedItems).some(item => item.id !== undefined);
    if (hasIds) {
      commands.push("selection-tools.copy-ids");
    }
    
    // Show duplication and deletion for multiple items
    if (itemCount > 1) {
      commands.push("selection-tools.duplicate", "selection-tools.delete");
    }
    
    return commands;
  }
});
```

### Advanced Examples

#### Node Alignment Tools
```javascript
app.registerExtension({
  name: "NodeAlignment",
  commands: [
    {
      id: "align.left",
      label: "Align Left",
      icon: "pi pi-align-left",
      function: () => {
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        if (selectedItems.length < 2) return;
        
        // Find leftmost position
        const leftmost = Math.min(...selectedItems.map(item => item.pos[0]));
        
        // Align all items to leftmost position
        selectedItems.forEach(item => {
          item.pos[0] = leftmost;
          item.setDirtyCanvas(true);
        });
        
        app.extensionManager.toast.add({
          severity: "success",
          summary: "Aligned",
          detail: "Nodes aligned to left",
          life: 2000
        });
      }
    },
    {
      id: "align.center",
      label: "Align Center",
      icon: "pi pi-align-center",
      function: () => {
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        if (selectedItems.length < 2) return;
        
        // Calculate center position
        const centerX = selectedItems.reduce((sum, item) => sum + item.pos[0], 0) / selectedItems.length;
        
        // Align all items to center
        selectedItems.forEach(item => {
          item.pos[0] = centerX;
          item.setDirtyCanvas(true);
        });
        
        app.extensionManager.toast.add({
          severity: "success",
          summary: "Aligned",
          detail: "Nodes aligned to center",
          life: 2000
        });
      }
    },
    {
      id: "align.distribute",
      label: "Distribute Evenly",
      icon: "pi pi-objects-column",
      function: () => {
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        if (selectedItems.length < 3) return;
        
        // Sort by X position
        selectedItems.sort((a, b) => a.pos[0] - b.pos[0]);
        
        // Calculate spacing
        const leftmost = selectedItems[0].pos[0];
        const rightmost = selectedItems[selectedItems.length - 1].pos[0];
        const spacing = (rightmost - leftmost) / (selectedItems.length - 1);
        
        // Distribute items
        selectedItems.forEach((item, index) => {
          item.pos[0] = leftmost + (spacing * index);
          item.setDirtyCanvas(true);
        });
        
        app.extensionManager.toast.add({
          severity: "success",
          summary: "Distributed",
          detail: "Nodes distributed evenly",
          life: 2000
        });
      }
    }
  ],
  
  getSelectionToolboxCommands: (selectedItem) => {
    const selectedItems = app.canvas.selectedItems;
    const itemCount = selectedItems ? selectedItems.size : 0;
    
    if (itemCount < 2) return [];
    
    const commands = ["align.left", "align.center"];
    
    if (itemCount >= 3) {
      commands.push("align.distribute");
    }
    
    return commands;
  }
});
```

#### Node Grouping Tools
```javascript
app.registerExtension({
  name: "NodeGrouping",
  commands: [
    {
      id: "group.create",
      label: "Create Group",
      icon: "pi pi-objects-column",
      function: () => {
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        if (selectedItems.length < 2) return;
        
        // Calculate group bounds
        const positions = selectedItems.map(item => item.pos);
        const minX = Math.min(...positions.map(pos => pos[0]));
        const minY = Math.min(...positions.map(pos => pos[1]));
        const maxX = Math.max(...positions.map(pos => pos[0]));
        const maxY = Math.max(...positions.map(pos => pos[1]));
        
        // Create group
        const group = {
          id: `group_${Date.now()}`,
          title: "New Group",
          pos: [minX, minY],
          size: [maxX - minX + 200, maxY - minY + 100],
          nodes: selectedItems.map(item => item.id)
        };
        
        console.log("Created group:", group);
        
        app.extensionManager.toast.add({
          severity: "success",
          summary: "Group Created",
          detail: `Grouped ${selectedItems.length} items`,
          life: 2000
        });
      }
    },
    {
      id: "group.ungroup",
      label: "Ungroup",
      icon: "pi pi-objects-column",
      function: () => {
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        const groups = selectedItems.filter(item => item.type === 'group');
        
        if (groups.length === 0) return;
        
        console.log(`Ungrouping ${groups.length} group${groups.length !== 1 ? 's' : ''}`);
        
        app.extensionManager.toast.add({
          severity: "success",
          summary: "Ungrouped",
          detail: `Ungrouped ${groups.length} group${groups.length !== 1 ? 's' : ''}`,
          life: 2000
        });
      }
    }
  ],
  
  getSelectionToolboxCommands: (selectedItem) => {
    const selectedItems = app.canvas.selectedItems;
    const itemCount = selectedItems ? selectedItems.size : 0;
    
    if (itemCount === 0) return [];
    
    const commands = [];
    
    // Show create group for multiple items
    if (itemCount >= 2) {
      commands.push("group.create");
    }
    
    // Show ungroup for groups
    const hasGroups = Array.from(selectedItems).some(item => item.type === 'group');
    if (hasGroups) {
      commands.push("group.ungroup");
    }
    
    return commands;
  }
});
```

### Best Practices

#### Command Design
```javascript
// Good: Clear, descriptive labels and appropriate icons
{
  id: "my-ext.align-nodes",
  label: "Align Nodes",
  icon: "pi pi-align-left",
  function: () => { /* ... */ }
}

// Bad: Vague, unclear labels
{
  id: "my-ext.action1",
  label: "Action",
  icon: "pi pi-circle",
  function: () => { /* ... */ }
}
```

#### Selection Handling
```javascript
// Proper selection validation
getSelectionToolboxCommands: (selectedItem) => {
  const selectedItems = app.canvas.selectedItems;
  const itemCount = selectedItems ? selectedItems.size : 0;
  
  // Early return for empty selection
  if (itemCount === 0) return [];
  
  // Validate selection types
  const hasNodes = Array.from(selectedItems).some(item => item.type && item.type !== 'group');
  const hasGroups = Array.from(selectedItems).some(item => item.type === 'group');
  
  const commands = [];
  
  if (hasNodes) {
    commands.push("node-commands");
  }
  
  if (hasGroups) {
    commands.push("group-commands");
  }
  
  return commands;
}
```

#### Performance Considerations
```javascript
// Efficient selection processing
getSelectionToolboxCommands: (selectedItem) => {
  const selectedItems = app.canvas.selectedItems;
  const itemCount = selectedItems ? selectedItems.size : 0;
  
  // Cache expensive operations
  if (itemCount > 0 && !this._cachedSelection) {
    this._cachedSelection = {
      count: itemCount,
      types: Array.from(selectedItems).map(item => item.type),
      hasIds: Array.from(selectedItems).some(item => item.id !== undefined)
    };
  }
  
  // Use cached data for command decisions
  if (this._cachedSelection) {
    const { count, types, hasIds } = this._cachedSelection;
    
    const commands = [];
    
    if (count > 1) {
      commands.push("multi-selection-command");
    }
    
    if (hasIds) {
      commands.push("id-based-command");
    }
    
    return commands;
  }
  
  return [];
}
```

### Important Notes

#### Settings Requirement
```javascript
// The selection toolbox must be enabled in settings
// Comfy.Canvas.SelectionToolbox must be enabled
```

#### Command Registration
```javascript
// Commands must be defined in the commands array before being referenced
app.registerExtension({
  name: "MyExtension",
  commands: [
    // Define commands first
    {
      id: "my-command",
      label: "My Command",
      function: () => { /* ... */ }
    }
  ],
  getSelectionToolboxCommands: (selectedItem) => {
    // Then reference them here
    return ["my-command"];
  }
});
```

#### Selection Access
```javascript
// Use app.canvas.selectedItems for all selected items
const selectedItems = app.canvas.selectedItems; // Set of all selected items

// For backward compatibility, app.canvas.selected_nodes still exists
const selectedNodes = app.canvas.selected_nodes; // Only nodes (deprecated)
}
```

## ComfyUI Commands and Keybindings - Keyboard Shortcuts

### Purpose
ComfyUI's Commands and Keybindings API allows extensions to register custom commands and associate them with keyboard shortcuts. This enables users to quickly trigger actions without using the mouse.

### Basic Usage

#### Simple Command and Keybinding Registration
```javascript
app.registerExtension({
  name: "MyExtension",
  // Register commands
  commands: [
    {
      id: "myCommand",
      label: "My Command",
      function: () => {
        console.log("Command executed!");
      }
    }
  ],
  // Associate keybindings with commands
  keybindings: [
    {
      combo: { key: "k", ctrl: true },
      commandId: "myCommand"
    }
  ]
});
```

**Features**:
- **Keyboard shortcuts**: Quick access to extension functions
- **Command system**: Standardized command interface
- **User productivity**: Faster workflow execution
- **Customizable**: Users can modify keybindings

### Command Configuration

#### Required Properties
```javascript
{
  id: string,              // Unique identifier for the command
  label: string,           // Display name for the command
  function: () => void     // Function to execute when command is triggered
}
```

**Properties**:
- **id**: Unique identifier for the command
- **label**: Display name for the command
- **function**: Function to execute when command is triggered

### Keybinding Configuration

#### Required Properties
```javascript
{
  combo: {                 // Key combination
    key: string,           // The main key (single character or special key)
    ctrl?: boolean,        // Require Ctrl key (optional)
    shift?: boolean,       // Require Shift key (optional)
    alt?: boolean,         // Require Alt key (optional)
    meta?: boolean         // Require Meta/Command key (optional)
  },
  commandId: string        // ID of the command to trigger
}
```

**Properties**:
- **combo**: Key combination object
- **commandId**: ID of the command to trigger
- **Modifier keys**: ctrl, shift, alt, meta (optional)

### Special Keys

#### Arrow Keys
```javascript
"ArrowUp"      // Up arrow
"ArrowDown"    // Down arrow
"ArrowLeft"    // Left arrow
"ArrowRight"   // Right arrow
```

#### Function Keys
```javascript
"F1" through "F12"  // Function keys F1-F12
```

#### Other Special Keys
```javascript
"Escape"       // Escape key
"Tab"          // Tab key
"Enter"        // Enter key
"Backspace"    // Backspace key
"Delete"       // Delete key
"Home"         // Home key
"End"          // End key
"PageUp"       // Page Up key
"PageDown"     // Page Down key
```

### Command Examples

#### Workflow Management Commands
```javascript
app.registerExtension({
  name: "WorkflowCommands",
  commands: [
    {
      id: "runWorkflow",
      label: "Run Workflow",
      function: () => {
        app.queuePrompt();
      }
    },
    {
      id: "clearWorkflow",
      label: "Clear Workflow",
      function: () => {
        if (confirm("Clear the workflow?")) {
          app.graph.clear();
        }
      }
    },
    {
      id: "saveWorkflow",
      label: "Save Workflow",
      function: () => {
        app.graphToPrompt().then(workflow => {
          const blob = new Blob([JSON.stringify(workflow)], {type: "application/json"});
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "workflow.json";
          a.click();
          URL.revokeObjectURL(url);
        });
      }
    },
    {
      id: "loadWorkflow",
      label: "Load Workflow",
      function: () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".json";
        input.onchange = (e) => {
          const file = e.target.files[0];
          if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
              try {
                const workflow = JSON.parse(e.target.result);
                app.loadGraphData(workflow);
              } catch (error) {
                console.error("Error loading workflow:", error);
              }
            };
            reader.readAsText(file);
          }
        };
        input.click();
      }
    }
  ]
});
```

#### Node Management Commands
```javascript
app.registerExtension({
  name: "NodeCommands",
  commands: [
    {
      id: "duplicateSelected",
      label: "Duplicate Selected",
      function: () => {
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        if (selectedItems.length > 0) {
          console.log(`Duplicating ${selectedItems.length} items`);
          // Implement duplication logic
        }
      }
    },
    {
      id: "deleteSelected",
      label: "Delete Selected",
      function: () => {
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        if (selectedItems.length > 0) {
          if (confirm(`Delete ${selectedItems.length} item${selectedItems.length !== 1 ? 's' : ''}?`)) {
            console.log(`Deleting ${selectedItems.length} items`);
            // Implement deletion logic
          }
        }
      }
    },
    {
      id: "selectAll",
      label: "Select All",
      function: () => {
        const nodes = app.graph._nodes;
        nodes.forEach(node => {
          node.setSelected(true);
        });
        app.canvas.setDirty(true);
      }
    },
    {
      id: "deselectAll",
      label: "Deselect All",
      function: () => {
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        selectedItems.forEach(item => {
          item.setSelected(false);
        });
        app.canvas.setDirty(true);
      }
    }
  ]
});
```

### Keybinding Examples

#### Workflow Keybindings
```javascript
app.registerExtension({
  name: "WorkflowKeybindings",
  commands: [
    /* Commands defined above */
  ],
  keybindings: [
    // Ctrl+R to run workflow
    {
      combo: { key: "r", ctrl: true },
      commandId: "runWorkflow"
    },
    // Ctrl+Shift+C to clear workflow
    {
      combo: { key: "c", ctrl: true, shift: true },
      commandId: "clearWorkflow"
    },
    // Ctrl+S to save workflow
    {
      combo: { key: "s", ctrl: true },
      commandId: "saveWorkflow"
    },
    // Ctrl+O to load workflow
    {
      combo: { key: "o", ctrl: true },
      commandId: "loadWorkflow"
    },
    // F5 to run workflow (alternative)
    {
      combo: { key: "F5" },
      commandId: "runWorkflow"
    }
  ]
});
```

#### Node Management Keybindings
```javascript
app.registerExtension({
  name: "NodeKeybindings",
  commands: [
    /* Commands defined above */
  ],
  keybindings: [
    // Ctrl+D to duplicate selected
    {
      combo: { key: "d", ctrl: true },
      commandId: "duplicateSelected"
    },
    // Delete key to delete selected
    {
      combo: { key: "Delete" },
      commandId: "deleteSelected"
    },
    // Ctrl+A to select all
    {
      combo: { key: "a", ctrl: true },
      commandId: "selectAll"
    },
    // Escape to deselect all
    {
      combo: { key: "Escape" },
      commandId: "deselectAll"
    }
  ]
});
```

### Advanced Examples

#### Extension Management Commands
```javascript
app.registerExtension({
  name: "ExtensionManager",
  commands: [
    {
      id: "toggleExtension",
      label: "Toggle Extension",
      function: () => {
        const extensionName = "MyExtension";
        const isEnabled = app.extensionManager.isEnabled(extensionName);
        
        if (isEnabled) {
          app.extensionManager.disable(extensionName);
          app.extensionManager.toast.add({
            severity: "info",
            summary: "Extension Disabled",
            detail: `${extensionName} has been disabled`,
            life: 3000
          });
        } else {
          app.extensionManager.enable(extensionName);
          app.extensionManager.toast.add({
            severity: "success",
            summary: "Extension Enabled",
            detail: `${extensionName} has been enabled`,
            life: 3000
          });
        }
      }
    },
    {
      id: "reloadExtensions",
      label: "Reload Extensions",
      function: () => {
        app.extensionManager.reload();
        app.extensionManager.toast.add({
          severity: "info",
          summary: "Extensions Reloaded",
          detail: "All extensions have been reloaded",
          life: 3000
        });
      }
    },
    {
      id: "showExtensionInfo",
      label: "Show Extension Info",
      function: () => {
        const extensions = app.extensionManager.getExtensions();
        console.log("Loaded extensions:", extensions);
        
        app.extensionManager.toast.add({
          severity: "info",
          summary: "Extension Info",
          detail: `${extensions.length} extensions loaded`,
          life: 3000
        });
      }
    }
  ],
  keybindings: [
    // Ctrl+Shift+E to toggle extension
    {
      combo: { key: "e", ctrl: true, shift: true },
      commandId: "toggleExtension"
    },
    // Ctrl+Shift+R to reload extensions
    {
      combo: { key: "r", ctrl: true, shift: true },
      commandId: "reloadExtensions"
    },
    // F12 to show extension info
    {
      combo: { key: "F12" },
      commandId: "showExtensionInfo"
    }
  ]
});
```

#### Canvas Navigation Commands
```javascript
app.registerExtension({
  name: "CanvasNavigation",
  commands: [
    {
      id: "zoomIn",
      label: "Zoom In",
      function: () => {
        app.canvas.zoomTo(1.2);
      }
    },
    {
      id: "zoomOut",
      label: "Zoom Out",
      function: () => {
        app.canvas.zoomTo(0.8);
      }
    },
    {
      id: "zoomFit",
      label: "Zoom to Fit",
      function: () => {
        app.canvas.zoomToFit();
      }
    },
    {
      id: "panLeft",
      label: "Pan Left",
      function: () => {
        const currentOffset = app.canvas.offset;
        app.canvas.offset = [currentOffset[0] - 100, currentOffset[1]];
      }
    },
    {
      id: "panRight",
      label: "Pan Right",
      function: () => {
        const currentOffset = app.canvas.offset;
        app.canvas.offset = [currentOffset[0] + 100, currentOffset[1]];
      }
    },
    {
      id: "panUp",
      label: "Pan Up",
      function: () => {
        const currentOffset = app.canvas.offset;
        app.canvas.offset = [currentOffset[0], currentOffset[1] - 100];
      }
    },
    {
      id: "panDown",
      label: "Pan Down",
      function: () => {
        const currentOffset = app.canvas.offset;
        app.canvas.offset = [currentOffset[0], currentOffset[1] + 100];
      }
    }
  ],
  keybindings: [
    // Ctrl+Plus to zoom in
    {
      combo: { key: "=", ctrl: true },
      commandId: "zoomIn"
    },
    // Ctrl+Minus to zoom out
    {
      combo: { key: "-", ctrl: true },
      commandId: "zoomOut"
    },
    // Ctrl+0 to zoom to fit
    {
      combo: { key: "0", ctrl: true },
      commandId: "zoomFit"
    },
    // Arrow keys for panning
    {
      combo: { key: "ArrowLeft" },
      commandId: "panLeft"
    },
    {
      combo: { key: "ArrowRight" },
      commandId: "panRight"
    },
    {
      combo: { key: "ArrowUp" },
      commandId: "panUp"
    },
    {
      combo: { key: "ArrowDown" },
      commandId: "panDown"
    }
  ]
});
```

### Best Practices

#### Command Design
```javascript
// Good: Clear, descriptive labels and unique IDs
{
  id: "my-extension.run-workflow",
  label: "Run Workflow",
  function: () => { /* ... */ }
}

// Bad: Vague, unclear labels
{
  id: "cmd1",
  label: "Action",
  function: () => { /* ... */ }
}
```

#### Keybinding Design
```javascript
// Good: Intuitive, memorable key combinations
{
  combo: { key: "r", ctrl: true },
  commandId: "runWorkflow"
}

// Bad: Confusing, hard to remember combinations
{
  combo: { key: "q", ctrl: true, shift: true, alt: true },
  commandId: "runWorkflow"
}
```

#### Error Handling
```javascript
{
  id: "safeCommand",
  label: "Safe Command",
  function: () => {
    try {
      // Perform potentially dangerous operation
      app.graph.clear();
      
      app.extensionManager.toast.add({
        severity: "success",
        summary: "Success",
        detail: "Operation completed successfully",
        life: 3000
      });
    } catch (error) {
      console.error("Command failed:", error);
      
      app.extensionManager.toast.add({
        severity: "error",
        summary: "Error",
        detail: "Operation failed: " + error.message,
        life: 5000
      });
    }
  }
}
```

#### User Confirmation
```javascript
{
  id: "destructiveCommand",
  label: "Destructive Command",
  function: () => {
    if (confirm("This action cannot be undone. Continue?")) {
      // Perform destructive operation
      console.log("Destructive operation executed");
    }
  }
}
```

### Important Notes and Limitations

#### Core Keybindings
```javascript
// Keybindings defined in ComfyUI core cannot be overwritten
// Check these source files for core keybindings:
// - Core Commands
// - Core Menu Commands  
// - Core Keybindings
```

#### Reserved Key Combinations
```javascript
// Some key combinations are reserved by the browser
// Examples of reserved combinations:
// - Ctrl+F: Browser search
// - Ctrl+R: Browser refresh
// - Ctrl+T: New tab
// - Ctrl+W: Close tab
// - Ctrl+N: New window
// - Ctrl+Shift+N: New incognito window
```

#### Extension Conflicts
```javascript
// If multiple extensions register the same keybinding, behavior is undefined
// Use unique key combinations to avoid conflicts
{
  combo: { key: "r", ctrl: true, shift: true },  // Unique combination
  commandId: "myUniqueCommand"
}
```

#### Keybinding Validation
```javascript
// Validate key combinations before registration
function validateKeybinding(combo) {
  const reserved = [
    { key: "f", ctrl: true },
    { key: "r", ctrl: true },
    { key: "t", ctrl: true },
    { key: "w", ctrl: true },
    { key: "n", ctrl: true }
  ];
  
  return !reserved.some(reserved => 
    reserved.key === combo.key && 
    reserved.ctrl === combo.ctrl
  );
}
```

## ComfyUI Topbar Menu - Custom Menu Items

### Purpose
ComfyUI's Topbar Menu API allows extensions to add custom menu items to the ComfyUI's top menu bar. This is useful for providing access to advanced features or less frequently used commands.

### Basic Usage

#### Simple Menu Registration
```javascript
app.registerExtension({
  name: "MyExtension",
  // Define commands
  commands: [
    { 
      id: "myCommand", 
      label: "My Command", 
      function: () => { alert("Command executed!"); } 
    }
  ],
  // Add commands to menu
  menuCommands: [
    { 
      path: ["Extensions", "My Extension"], 
      commands: ["myCommand"] 
    }
  ]
});
```

**Features**:
- **Menu integration**: Add items to ComfyUI's top menu bar
- **Command system**: Uses the same command system as keybindings
- **Menu hierarchy**: Create nested menu structures
- **Multiple locations**: Add commands to multiple menu locations

### Command Configuration

#### Required Properties
```javascript
{
  id: string,              // Unique identifier for the command
  label: string,           // Display name for the command
  function: () => void     // Function to execute when command is triggered
}
```

**Properties**:
- **id**: Unique identifier for the command
- **label**: Display name for the command
- **function**: Function to execute when command is triggered

### Menu Configuration

#### Required Properties
```javascript
{
  path: string[],          // Array representing menu hierarchy
  commands: string[]       // Array of command IDs to add at this location
}
```

**Properties**:
- **path**: Array representing menu hierarchy
- **commands**: Array of command IDs to add at this location

### Menu Examples

#### Adding to Existing Menus
```javascript
app.registerExtension({
  name: "MenuExamples",
  commands: [
    { 
      id: "saveAsImage", 
      label: "Save as Image", 
      function: () => { 
        // Code to save canvas as image
        const canvas = app.canvas.canvas;
        const dataURL = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = 'workflow.png';
        link.href = dataURL;
        link.click();
      } 
    },
    { 
      id: "exportWorkflow", 
      label: "Export Workflow", 
      function: () => { 
        // Code to export workflow
        app.graphToPrompt().then(workflow => {
          const blob = new Blob([JSON.stringify(workflow)], {type: "application/json"});
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "workflow.json";
          a.click();
          URL.revokeObjectURL(url);
        });
      } 
    },
    { 
      id: "importWorkflow", 
      label: "Import Workflow", 
      function: () => { 
        // Code to import workflow
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".json";
        input.onchange = (e) => {
          const file = e.target.files[0];
          if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
              try {
                const workflow = JSON.parse(e.target.result);
                app.loadGraphData(workflow);
              } catch (error) {
                console.error("Error loading workflow:", error);
              }
            };
            reader.readAsText(file);
          }
        };
        input.click();
      } 
    }
  ],
  menuCommands: [
    // Add to File menu
    { 
      path: ["File"], 
      commands: ["saveAsImage", "exportWorkflow", "importWorkflow"] 
    }
  ]
});
```

#### Creating Submenu Structure
```javascript
app.registerExtension({
  name: "SubmenuExample",
  commands: [
    { 
      id: "option1", 
      label: "Option 1", 
      function: () => { console.log("Option 1"); } 
    },
    { 
      id: "option2", 
      label: "Option 2", 
      function: () => { console.log("Option 2"); } 
    },
    { 
      id: "suboption1", 
      label: "Sub-option 1", 
      function: () => { console.log("Sub-option 1"); } 
    },
    { 
      id: "suboption2", 
      label: "Sub-option 2", 
      function: () => { console.log("Sub-option 2"); } 
    },
    { 
      id: "advancedOption", 
      label: "Advanced Option", 
      function: () => { console.log("Advanced Option"); } 
    }
  ],
  menuCommands: [
    // Create a nested menu structure
    { 
      path: ["Extensions", "My Tools"], 
      commands: ["option1", "option2"] 
    },
    { 
      path: ["Extensions", "My Tools", "Advanced"], 
      commands: ["suboption1", "suboption2"] 
    },
    { 
      path: ["Extensions", "My Tools", "Advanced", "Expert"], 
      commands: ["advancedOption"] 
    }
  ]
});
```

#### Multiple Menu Locations
```javascript
app.registerExtension({
  name: "MultiLocationExample",
  commands: [
    { 
      id: "helpCommand", 
      label: "Get Help", 
      function: () => { window.open("https://docs.example.com", "_blank"); } 
    },
    { 
      id: "documentation", 
      label: "Documentation", 
      function: () => { window.open("https://docs.example.com", "_blank"); } 
    },
    { 
      id: "support", 
      label: "Support", 
      function: () => { window.open("https://support.example.com", "_blank"); } 
    }
  ],
  menuCommands: [
    // Add to Help menu
    { 
      path: ["Help"], 
      commands: ["helpCommand", "documentation", "support"] 
    },
    // Also add to Extensions menu
    { 
      path: ["Extensions"], 
      commands: ["helpCommand"] 
    }
  ]
});
```

### Advanced Examples

#### Workflow Management Menu
```javascript
app.registerExtension({
  name: "WorkflowManager",
  commands: [
    { 
      id: "newWorkflow", 
      label: "New Workflow", 
      function: () => { 
        if (confirm("Create a new workflow? Unsaved changes will be lost.")) {
          app.graph.clear();
        }
      } 
    },
    { 
      id: "saveWorkflow", 
      label: "Save Workflow", 
      function: () => { 
        app.graphToPrompt().then(workflow => {
          const blob = new Blob([JSON.stringify(workflow)], {type: "application/json"});
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "workflow.json";
          a.click();
          URL.revokeObjectURL(url);
        });
      } 
    },
    { 
      id: "loadWorkflow", 
      label: "Load Workflow", 
      function: () => { 
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".json";
        input.onchange = (e) => {
          const file = e.target.files[0];
          if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
              try {
                const workflow = JSON.parse(e.target.result);
                app.loadGraphData(workflow);
              } catch (error) {
                console.error("Error loading workflow:", error);
              }
            };
            reader.readAsText(file);
          }
        };
        input.click();
      } 
    },
    { 
      id: "runWorkflow", 
      label: "Run Workflow", 
      function: () => { 
        app.queuePrompt();
      } 
    },
    { 
      id: "clearWorkflow", 
      label: "Clear Workflow", 
      function: () => { 
        if (confirm("Clear the workflow?")) {
          app.graph.clear();
        }
      } 
    }
  ],
  menuCommands: [
    { 
      path: ["File"], 
      commands: ["newWorkflow", "saveWorkflow", "loadWorkflow"] 
    },
    { 
      path: ["Run"], 
      commands: ["runWorkflow", "clearWorkflow"] 
    }
  ]
});
```

#### Node Management Menu
```javascript
app.registerExtension({
  name: "NodeManager",
  commands: [
    { 
      id: "selectAll", 
      label: "Select All", 
      function: () => { 
        const nodes = app.graph._nodes;
        nodes.forEach(node => {
          node.setSelected(true);
        });
        app.canvas.setDirty(true);
      } 
    },
    { 
      id: "deselectAll", 
      label: "Deselect All", 
      function: () => { 
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        selectedItems.forEach(item => {
          item.setSelected(false);
        });
        app.canvas.setDirty(true);
      } 
    },
    { 
      id: "duplicateSelected", 
      label: "Duplicate Selected", 
      function: () => { 
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        if (selectedItems.length > 0) {
          console.log(`Duplicating ${selectedItems.length} items`);
          // Implement duplication logic
        }
      } 
    },
    { 
      id: "deleteSelected", 
      label: "Delete Selected", 
      function: () => { 
        const selectedItems = Array.from(app.canvas.selectedItems || []);
        if (selectedItems.length > 0) {
          if (confirm(`Delete ${selectedItems.length} item${selectedItems.length !== 1 ? 's' : ''}?`)) {
            console.log(`Deleting ${selectedItems.length} items`);
            // Implement deletion logic
          }
        }
      } 
    }
  ],
  menuCommands: [
    { 
      path: ["Edit"], 
      commands: ["selectAll", "deselectAll", "duplicateSelected", "deleteSelected"] 
    }
  ]
});
```

#### Extension Management Menu
```javascript
app.registerExtension({
  name: "ExtensionManager",
  commands: [
    { 
      id: "reloadExtensions", 
      label: "Reload Extensions", 
      function: () => { 
        app.extensionManager.reload();
        app.extensionManager.toast.add({
          severity: "info",
          summary: "Extensions Reloaded",
          detail: "All extensions have been reloaded",
          life: 3000
        });
      } 
    },
    { 
      id: "showExtensionInfo", 
      label: "Show Extension Info", 
      function: () => { 
        const extensions = app.extensionManager.getExtensions();
        console.log("Loaded extensions:", extensions);
        
        app.extensionManager.toast.add({
          severity: "info",
          summary: "Extension Info",
          detail: `${extensions.length} extensions loaded`,
          life: 3000
        });
      } 
    },
    { 
      id: "toggleExtension", 
      label: "Toggle Extension", 
      function: () => { 
        const extensionName = "MyExtension";
        const isEnabled = app.extensionManager.isEnabled(extensionName);
        
        if (isEnabled) {
          app.extensionManager.disable(extensionName);
          app.extensionManager.toast.add({
            severity: "info",
            summary: "Extension Disabled",
            detail: `${extensionName} has been disabled`,
            life: 3000
          });
        } else {
          app.extensionManager.enable(extensionName);
          app.extensionManager.toast.add({
            severity: "success",
            summary: "Extension Enabled",
            detail: `${extensionName} has been enabled`,
            life: 3000
          });
        }
      } 
    }
  ],
  menuCommands: [
    { 
      path: ["Extensions"], 
      commands: ["reloadExtensions", "showExtensionInfo", "toggleExtension"] 
    }
  ]
});
```

#### Canvas Navigation Menu
```javascript
app.registerExtension({
  name: "CanvasNavigation",
  commands: [
    { 
      id: "zoomIn", 
      label: "Zoom In", 
      function: () => { 
        app.canvas.zoomTo(1.2);
      } 
    },
    { 
      id: "zoomOut", 
      label: "Zoom Out", 
      function: () => { 
        app.canvas.zoomTo(0.8);
      } 
    },
    { 
      id: "zoomFit", 
      label: "Zoom to Fit", 
      function: () => { 
        app.canvas.zoomToFit();
      } 
    },
    { 
      id: "resetView", 
      label: "Reset View", 
      function: () => { 
        app.canvas.zoomTo(1.0);
        app.canvas.offset = [0, 0];
      } 
    }
  ],
  menuCommands: [
    { 
      path: ["View"], 
      commands: ["zoomIn", "zoomOut", "zoomFit", "resetView"] 
    }
  ]
});
```

### Best Practices

#### Menu Organization
```javascript
// Good: Logical menu organization
menuCommands: [
  { 
    path: ["File"], 
    commands: ["saveWorkflow", "loadWorkflow"] 
  },
  { 
    path: ["Edit"], 
    commands: ["selectAll", "deselectAll"] 
  },
  { 
    path: ["View"], 
    commands: ["zoomIn", "zoomOut"] 
  }
]

// Bad: Unclear menu organization
menuCommands: [
  { 
    path: ["File"], 
    commands: ["zoomIn", "selectAll", "saveWorkflow"] 
  }
]
```

#### Command Naming
```javascript
// Good: Clear, descriptive command names
{ 
  id: "save-workflow-as-json", 
  label: "Save Workflow as JSON", 
  function: () => { /* ... */ } 
}

// Bad: Vague, unclear command names
{ 
  id: "save1", 
  label: "Save", 
  function: () => { /* ... */ } 
}
```

#### Error Handling
```javascript
{ 
  id: "safeCommand", 
  label: "Safe Command", 
  function: () => { 
    try {
      // Perform potentially dangerous operation
      app.graph.clear();
      
      app.extensionManager.toast.add({
        severity: "success",
        summary: "Success",
        detail: "Operation completed successfully",
        life: 3000
      });
    } catch (error) {
      console.error("Command failed:", error);
      
      app.extensionManager.toast.add({
        severity: "error",
        summary: "Error",
        detail: "Operation failed: " + error.message,
        life: 5000
      });
    }
  } 
}
```

#### User Confirmation
```javascript
{ 
  id: "destructiveCommand", 
  label: "Destructive Command", 
  function: () => { 
    if (confirm("This action cannot be undone. Continue?")) {
      // Perform destructive operation
      console.log("Destructive operation executed");
    }
  } 
}
```

### Integration with Other APIs

#### Settings Integration
```javascript
app.registerExtension({
  name: "SettingsIntegration",
  commands: [
    { 
      id: "openSettings", 
      label: "Open Settings", 
      function: () => { 
        // Open settings panel
        app.settings.show();
      } 
    },
    { 
      id: "resetSettings", 
      label: "Reset Settings", 
      function: () => { 
        if (confirm("Reset all settings to default?")) {
          app.settings.reset();
        }
      } 
    }
  ],
  menuCommands: [
    { 
      path: ["Settings"], 
      commands: ["openSettings", "resetSettings"] 
    }
  ]
});
```

#### Dialog Integration
```javascript
app.registerExtension({
  name: "DialogIntegration",
  commands: [
    { 
      id: "showDialog", 
      label: "Show Dialog", 
      function: () => { 
        app.dialog.prompt({
          title: "Custom Dialog",
          message: "Enter your input:",
          onConfirm: (value) => {
            console.log("User input:", value);
          }
        });
      } 
    }
  ],
  menuCommands: [
    { 
      path: ["Tools"], 
      commands: ["showDialog"] 
    }
  ]
});
```

### Important Notes

#### Menu Hierarchy
```javascript
// Menu paths are hierarchical
// ["File"] -> File menu
// ["File", "Export"] -> Export submenu under File
// ["Extensions", "My Tools"] -> My Tools submenu under Extensions
```

#### Command Reuse
```javascript
// Commands can be used in multiple menu locations
menuCommands: [
  { 
    path: ["File"], 
    commands: ["saveWorkflow"] 
  },
  { 
    path: ["Edit"], 
    commands: ["saveWorkflow"] 
  }
]
```

#### Menu Ordering
```javascript
// Menu items appear in the order they are defined
// Use consistent ordering for better user experience
}
```

## ComfyUI Annotated Examples - UI Interactions

### Purpose
ComfyUI's Annotated Examples provide practical code fragments for common UI interactions, event handling, and menu customization. This collection demonstrates how to extend ComfyUI's interface with custom functionality.

### Right Click Menus

#### Background Menu
The main background menu (right-click on the canvas) is generated by a call to `LGraph.getCanvasMenuOptions`. One way to add your own menu options is to hijack this call:

```javascript
/* in setup() */
const original_getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
LGraphCanvas.prototype.getCanvasMenuOptions = function () {
    // get the basic options 
    const options = original_getCanvasMenuOptions.apply(this, arguments);
    options.push(null); // inserts a divider
    options.push({
        content: "The text for the menu",
        callback: async () => {
            // do whatever
        }
    })
    return options;
}
```

**Features**:
- **Menu hijacking**: Override default canvas menu options
- **Divider support**: Add visual separators with `null`
- **Custom callbacks**: Execute custom functions on menu selection
- **Canvas integration**: Seamless integration with canvas context

#### Node Menu
When you right click on a node, the menu is similarly generated by `node.getExtraMenuOptions`. But instead of returning an options object, this one gets it passed in:

```javascript
/* in beforeRegisterNodeDef() */
if (nodeType?.comfyClass=="MyNodeClass") { 
    const original_getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function(_, options) {
        original_getExtraMenuOptions?.apply(this, arguments);
        options.push({
            content: "Do something fun",
            callback: async () => {
                // fun thing
            }
        })
    }   
}
```

**Features**:
- **Node-specific menus**: Custom menus for specific node types
- **Menu extension**: Add options to existing node menus
- **Node context**: Access to node instance in callbacks
- **Type safety**: Check node type before adding menu options

#### Submenus
If you want a submenu, provide a callback which uses `LiteGraph.ContextMenu` to create it:

```javascript
function make_submenu(value, options, e, menu, node) {
    const submenu = new LiteGraph.ContextMenu(
        ["option 1", "option 2", "option 3"],
        { 
            event: e, 
            callback: function (v) { 
                // do something with v (=="option x")
            }, 
            parentMenu: menu, 
            node:node
        }
    )
}

/* ... */
    options.push(
        {
            content: "Menu with options",
            has_submenu: true,
            callback: make_submenu,
        }
    )
```

**Features**:
- **Nested menus**: Create submenu structures
- **Context preservation**: Maintain node and event context
- **Dynamic options**: Generate menu options dynamically
- **Parent menu integration**: Seamless submenu integration

### Capture UI Events

#### Button Click Detection
This works just like you'd expect - find the UI element in the DOM and add an eventListener. `setup()` is a good place to do this, since the page has fully loaded:

```javascript
function queue_button_pressed() { 
    console.log("Queue button was pressed!") 
}
document.getElementById("queue-button").addEventListener("click", queue_button_pressed);
```

**Features**:
- **DOM event handling**: Standard DOM event listeners
- **Button detection**: Capture button clicks
- **Setup timing**: Use `setup()` for proper initialization
- **Event context**: Access to event object and target

#### Workflow Events
This is one of many API events:

```javascript
import { api } from "../../scripts/api.js";
/* in setup() */
function on_execution_start() { 
    /* do whatever */
}
api.addEventListener("execution_start", on_execution_start);
```

**Available Events**:
- **execution_start**: Workflow execution begins
- **execution_cached**: Workflow execution cached
- **execution_success**: Workflow execution successful
- **execution_error**: Workflow execution failed
- **execution_interrupted**: Workflow execution interrupted

#### API Hijacking
A simple example of hijacking the API:

```javascript
import { api } from "../../scripts/api.js";
/* in setup() */
const original_api_interrupt = api.interrupt;
api.interrupt = function () {
    /* Do something before the original method is called */
    original_api_interrupt.apply(this, arguments);
    /* Or after */
}
```

**Features**:
- **Method hijacking**: Override API methods
- **Before/after execution**: Execute code before or after original method
- **Argument preservation**: Maintain original method arguments
- **Context preservation**: Maintain `this` context

#### Node Click Detection
Node has a `mouseDown` method you can hijack. This time we're careful to pass on any return value:

```javascript
async nodeCreated(node) {
    if (node?.comfyClass === "My Node Name") {
        const original_onMouseDown = node.onMouseDown;
        node.onMouseDown = function( e, pos, canvas ) {
            alert("ouch!");
            return original_onMouseDown?.apply(this, arguments);
        }        
    }
}
```

**Features**:
- **Node-specific events**: Custom behavior for specific node types
- **Mouse interaction**: Capture mouse events on nodes
- **Return value preservation**: Maintain original method return values
- **Event context**: Access to event, position, and canvas

### Advanced Examples

#### Custom Canvas Menu
```javascript
app.registerExtension({
  name: "CustomCanvasMenu",
  setup() {
    const original_getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
      const options = original_getCanvasMenuOptions.apply(this, arguments);
      
      // Add divider
      options.push(null);
      
      // Add custom options
      options.push({
        content: "Save Canvas as Image",
        callback: async () => {
          const canvas = this.canvas;
          const dataURL = canvas.toDataURL('image/png');
          const link = document.createElement('a');
          link.download = 'canvas.png';
          link.href = dataURL;
          link.click();
        }
      });
      
      options.push({
        content: "Clear Canvas",
        callback: async () => {
          if (confirm("Clear the canvas?")) {
            this.graph.clear();
          }
        }
      });
      
      return options;
    };
  }
});
```

#### Custom Node Menu
```javascript
app.registerExtension({
  name: "CustomNodeMenu",
  beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeType?.comfyClass === "MyCustomNode") {
      const original_getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
      nodeType.prototype.getExtraMenuOptions = function(_, options) {
        original_getExtraMenuOptions?.apply(this, arguments);
        
        options.push({
          content: "Duplicate Node",
          callback: async () => {
            const newNode = this.clone();
            newNode.pos = [this.pos[0] + 50, this.pos[1] + 50];
            this.graph.add(newNode);
          }
        });
        
        options.push({
          content: "Node Info",
          callback: async () => {
            alert(`Node: ${this.title}\nType: ${this.type}\nPosition: ${this.pos}`);
          }
        });
      };
    }
  }
});
```

#### Submenu Example
```javascript
app.registerExtension({
  name: "SubmenuExample",
  setup() {
    const original_getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
      const options = original_getCanvasMenuOptions.apply(this, arguments);
      
      function createExportSubmenu(value, options, e, menu, node) {
        const submenu = new LiteGraph.ContextMenu(
          ["Export as JSON", "Export as PNG", "Export as SVG"],
          { 
            event: e, 
            callback: function (v) { 
              switch(v) {
                case "Export as JSON":
                  // Export workflow as JSON
                  app.graphToPrompt().then(workflow => {
                    const blob = new Blob([JSON.stringify(workflow)], {type: "application/json"});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = "workflow.json";
                    a.click();
                    URL.revokeObjectURL(url);
                  });
                  break;
                case "Export as PNG":
                  // Export canvas as PNG
                  const canvas = this.canvas;
                  const dataURL = canvas.toDataURL('image/png');
                  const link = document.createElement('a');
                  link.download = 'workflow.png';
                  link.href = dataURL;
                  link.click();
                  break;
                case "Export as SVG":
                  // Export as SVG (placeholder)
                  alert("SVG export not implemented yet");
                  break;
              }
            }, 
            parentMenu: menu, 
            node: node
          }
        );
      }
      
      options.push({
        content: "Export",
        has_submenu: true,
        callback: createExportSubmenu,
      });
      
      return options;
    };
  }
});
```

#### Event Detection Examples
```javascript
app.registerExtension({
  name: "EventDetection",
  setup() {
    // Detect queue button clicks
    const queueButton = document.getElementById("queue-button");
    if (queueButton) {
      queueButton.addEventListener("click", () => {
        console.log("Queue button clicked!");
      });
    }
    
    // Detect workflow events
    import { api } from "../../scripts/api.js";
    
    api.addEventListener("execution_start", () => {
      console.log("Workflow execution started");
    });
    
    api.addEventListener("execution_success", () => {
      console.log("Workflow execution completed successfully");
    });
    
    api.addEventListener("execution_error", (error) => {
      console.log("Workflow execution failed:", error);
    });
    
    // Hijack API methods
    const original_interrupt = api.interrupt;
    api.interrupt = function () {
      console.log("Workflow interrupted by user");
      return original_interrupt.apply(this, arguments);
    };
  }
});
```

#### Node Event Handling
```javascript
app.registerExtension({
  name: "NodeEventHandling",
  nodeCreated(node) {
    if (node?.comfyClass === "MyCustomNode") {
      // Hijack mouse down
      const original_onMouseDown = node.onMouseDown;
      node.onMouseDown = function(e, pos, canvas) {
        console.log("Node clicked:", this.title);
        return original_onMouseDown?.apply(this, arguments);
      };
      
      // Hijack mouse up
      const original_onMouseUp = node.onMouseUp;
      node.onMouseUp = function(e, pos, canvas) {
        console.log("Node released:", this.title);
        return original_onMouseUp?.apply(this, arguments);
      };
      
      // Hijack mouse move
      const original_onMouseMove = node.onMouseMove;
      node.onMouseMove = function(e, pos, canvas) {
        // Only log occasionally to avoid spam
        if (Math.random() < 0.01) {
          console.log("Node moved:", this.title);
        }
        return original_onMouseMove?.apply(this, arguments);
      };
    }
  }
});
```

### Best Practices

#### Menu Organization
```javascript
// Good: Clear menu structure with dividers
options.push(null); // Divider
options.push({
  content: "Clear Canvas",
  callback: async () => { /* ... */ }
});

// Bad: No organization
options.push({
  content: "Clear Canvas",
  callback: async () => { /* ... */ }
});
```

#### Event Handling
```javascript
// Good: Proper event listener setup
function setupEventListeners() {
  const button = document.getElementById("my-button");
  if (button) {
    button.addEventListener("click", handleClick);
  }
}

// Bad: No null checking
document.getElementById("my-button").addEventListener("click", handleClick);
```

#### Method Hijacking
```javascript
// Good: Preserve original method behavior
const original_method = obj.method;
obj.method = function() {
  // Do something before
  const result = original_method.apply(this, arguments);
  // Do something after
  return result;
};

// Bad: Don't preserve original behavior
obj.method = function() {
  // Only custom behavior, original method lost
};
```

#### Error Handling
```javascript
// Good: Safe event handling
function safeEventHandler() {
  try {
    // Potentially dangerous operation
    app.graph.clear();
  } catch (error) {
    console.error("Error in event handler:", error);
  }
}

// Bad: No error handling
function unsafeEventHandler() {
  app.graph.clear(); // Could throw error
}
```

### Important Notes

#### Timing Considerations
```javascript
// setup() is called after page load
app.registerExtension({
  name: "MyExtension",
  setup() {
    // Safe to access DOM elements here
    const button = document.getElementById("my-button");
    if (button) {
      button.addEventListener("click", handleClick);
    }
  }
});
```

#### Method Preservation
```javascript
// Always preserve original method behavior
const original_method = obj.method;
obj.method = function() {
  // Custom behavior
  return original_method.apply(this, arguments);
};
```

#### Event Context
```javascript
// Access event context in callbacks
function handleClick(event) {
  console.log("Clicked element:", event.target);
  console.log("Event type:", event.type);
  console.log("Mouse position:", event.clientX, event.clientY);
}
```

## ComfyUI i18n Support - Multi-Language Custom Nodes

### Purpose
ComfyUI supports internationalization (i18n) for custom nodes, allowing developers to provide multi-language support for their extensions. This enables users to interact with nodes in their preferred language.

### Supported Languages
ComfyUI currently supports the following languages:
- **English (en)** - Default/base language
- **Chinese (Simplified) (zh)** - 简体中文
- **Chinese (Traditional) (zh-TW)** - 繁體中文
- **French (fr)** - Français
- **Korean (ko)** - 한국어
- **Russian (ru)** - Русский
- **Spanish (es)** - Español
- **Japanese (ja)** - 日本語
- **Arabic (ar)** - العربية

### Directory Structure
Create a `locales` folder under your custom node with the following structure:

```
your_custom_node/
├── __init__.py
├── your_node.py
└── locales/                   # i18n support folder
    ├── en/                    # English translations (recommended as base)
    │   ├── main.json          # General English translation content
    │   ├── nodeDefs.json      # English node definition translations
    │   ├── settings.json      # Optional: settings interface translations
    │   └── commands.json      # Optional: command translations
    ├── zh/                    # Chinese translation files
    │   ├── nodeDefs.json      # Chinese node definition translations
    │   ├── main.json          # General Chinese translation content
    │   ├── settings.json      # Chinese settings interface translations
    │   └── commands.json      # Chinese command translations
    ├── fr/                    # French translation files
    │   ├── nodeDefs.json
    │   ├── main.json
    │   └── settings.json
    └── ...                    # Other language folders
```

### Node Definition Translation (nodeDefs.json)

#### Example Node Class
```python
class I18nTextProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello World!",
                    "tooltip": "The original text content to be processed"
                }),
                "operation": (["uppercase", "lowercase", "reverse", "add_prefix"], {
                    "default": "uppercase",
                    "tooltip": "The text processing operation to be executed"
                }),
                "count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "The number of times to repeat the operation"
                }),
            },
            "optional": {
                "prefix": ("STRING", {
                    "default": "[I18N] ",
                    "multiline": False,
                    "tooltip": "The prefix to add to the text"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "process_text"
    CATEGORY = "I18n Demo"
    DESCRIPTION = "A simple i18n demo node that demonstrates text processing with internationalization support"

    def process_text(self, text, operation, count, prefix=""):
        try:
            result = text
            
            for _ in range(count):
                if operation == "uppercase":
                    result = result.upper()
                elif operation == "lowercase":
                    result = result.lower()
                elif operation == "reverse":
                    result = result[::-1]
                elif operation == "add_prefix":
                    result = prefix + result
            
            return (result,)
        except Exception as e:
            print(f"I18nTextProcessor error: {e}")
            return (f"Error: {str(e)}",)
```

#### Corresponding nodeDefs.json
```json
{
  "I18nTextProcessor": {
    "display_name": "I18n Text Processor",
    "description": "A simple i18n demo node that demonstrates text processing with internationalization support",
    "inputs": {
      "text": {
        "name": "Text Input",
        "tooltip": "The original text content to be processed"
      },
      "operation": {
        "name": "Operation Type",
        "tooltip": "The text processing operation to be executed",
        "options": {
          "uppercase": "To Uppercase",
          "lowercase": "To Lowercase",
          "reverse": "Reverse Text",
          "add_prefix": "Add Prefix"
        }
      },
      "count": {
        "name": "Repeat Count",
        "tooltip": "The number of times to repeat the operation"
      },
      "prefix": {
        "name": "Prefix Text",
        "tooltip": "The prefix to add to the text"
      }
    },
    "outputs": {
      "0": {
        "name": "Processed Text",
        "tooltip": "The final processed text result"
      }
    }
  }
}
```

**Key Points**:
- **Output indexing**: Use numeric indices (0, 1, 2...) for outputs instead of names
- **Option translation**: Translate dropdown options in the `options` object
- **Tooltip translation**: Translate all tooltip text for better user experience

### Menu Settings Translation

#### Extension Registration
```javascript
app.registerExtension({
    name: "I18nDemo",
    settings: [
        {
            id: "I18nDemo.EnableDebugMode",
            category: ["I18nDemo","DebugMode"], // Matches settingsCategories key in main.json
            name: "Enable Debug Mode", // Will be overridden by translation
            tooltip: "Show debug information in console for i18n demo nodes", // Will be overridden by translation
            type: "boolean",
            defaultValue: false,
            experimental: true,
            onChange: (value) => {
                console.log("I18n Demo:", value ? "Debug mode enabled" : "Debug mode disabled");
            }
        },
        {
            id: "I18nDemo.DefaultTextOperation",
            category: ["I18nDemo","DefaultTextOperation"], // Matches settingsCategories key in main.json
            name: "Default Text Operation", // Will be overridden by translation
            tooltip: "Default operation for text processor node", // Will be overridden by translation
            type: "combo",
            options: ["uppercase", "lowercase", "reverse", "add_prefix"],
            defaultValue: "uppercase",
            experimental: true
        }
    ],
})
```

#### Settings Categories (main.json)
```json
{
  "settingsCategories": {
    "I18nDemo": "I18n Demo",
    "DebugMode": "Debug Mode",
    "DefaultTextOperation": "Default Text Operation"
  }
}
```

#### Settings Translation (settings.json)
```json
{
  "I18nDemo_EnableDebugMode": {
    "name": "Enable Debug Mode",
    "tooltip": "Show debug information in console for i18n demo nodes"
  },
  "I18nDemo_DefaultTextOperation": {
    "name": "Default Text Operation",
    "tooltip": "Default operation for text processor node",
    "options": {
      "uppercase": "Uppercase",
      "lowercase": "Lowercase",
      "reverse": "Reverse",
      "add_prefix": "Add Prefix"
    }
  }
}
```

**Translation Key Rules**:
- Replace dots (.) with underscores (_) in setting IDs
- Example: `"I18nDemo.EnableDebugMode"` becomes `"I18nDemo_EnableDebugMode"`

### Translation File Examples

#### English (en/main.json)
```json
{
  "settingsCategories": {
    "I18nDemo": "I18n Demo",
    "DebugMode": "Debug Mode",
    "DefaultTextOperation": "Default Text Operation"
  },
  "general": {
    "welcome": "Welcome to I18n Demo",
    "description": "This extension demonstrates internationalization support"
  }
}
```

#### Chinese (zh/main.json)
```json
{
  "settingsCategories": {
    "I18nDemo": "国际化演示",
    "DebugMode": "调试模式",
    "DefaultTextOperation": "默认文本操作"
  },
  "general": {
    "welcome": "欢迎使用国际化演示",
    "description": "此扩展演示了国际化支持功能"
  }
}
```

#### French (fr/main.json)
```json
{
  "settingsCategories": {
    "I18nDemo": "Démo I18n",
    "DebugMode": "Mode Debug",
    "DefaultTextOperation": "Opération de Texte par Défaut"
  },
  "general": {
    "welcome": "Bienvenue dans la démo I18n",
    "description": "Cette extension démontre le support de l'internationalisation"
  }
}
```

### Best Practices

#### Translation Key Naming
```json
// Good: Clear, hierarchical naming
{
  "I18nDemo_EnableDebugMode": {
    "name": "Enable Debug Mode",
    "tooltip": "Show debug information in console"
  }
}

// Bad: Unclear naming
{
  "setting1": {
    "name": "Setting 1",
    "tooltip": "Some setting"
  }
}
```

#### Consistent Translation
```json
// Good: Consistent terminology
{
  "uppercase": "Uppercase",
  "lowercase": "Lowercase",
  "reverse": "Reverse"
}

// Bad: Inconsistent terminology
{
  "uppercase": "UPPERCASE",
  "lowercase": "lower case",
  "reverse": "REVERSE"
}
```

#### Complete Translation Coverage
```json
// Good: All text translated
{
  "I18nTextProcessor": {
    "display_name": "I18n Text Processor",
    "description": "A simple i18n demo node...",
    "inputs": {
      "text": {
        "name": "Text Input",
        "tooltip": "The original text content..."
      }
    }
  }
}

// Bad: Missing translations
{
  "I18nTextProcessor": {
    "display_name": "I18n Text Processor",
    "description": "A simple i18n demo node...",
    "inputs": {
      "text": {
        "name": "Text Input"
        // Missing tooltip translation
      }
    }
  }
}
```

### Advanced Features

#### Dynamic Translation Loading
```javascript
// Load translations dynamically
function loadTranslations(language) {
  const translations = {};
  
  try {
    const response = fetch(`/extensions/I18nDemo/locales/${language}/main.json`);
    const data = response.json();
    Object.assign(translations, data);
  } catch (error) {
    console.warn(`Failed to load ${language} translations:`, error);
  }
  
  return translations;
}
```

#### Fallback Translation
```javascript
// Implement fallback to English if translation missing
function getTranslation(key, language = 'en') {
  const translations = loadTranslations(language);
  
  if (translations[key]) {
    return translations[key];
  }
  
  // Fallback to English
  if (language !== 'en') {
    const englishTranslations = loadTranslations('en');
    return englishTranslations[key] || key;
  }
  
  return key;
}
```

#### Context-Aware Translation
```javascript
// Translation with context
function translateWithContext(key, context = {}) {
  const translation = getTranslation(key);
  
  // Replace placeholders
  return translation.replace(/\{(\w+)\}/g, (match, placeholder) => {
    return context[placeholder] || match;
  });
}

// Usage
const message = translateWithContext('welcome_user', { 
  username: 'John',
  language: 'English'
});
```

### Important Notes

#### File Encoding
```json
// Always use UTF-8 encoding for translation files
// This ensures proper display of special characters
{
  "chinese_text": "你好世界",
  "french_text": "Bonjour le monde",
  "arabic_text": "مرحبا بالعالم"
}
```

#### Translation Validation
```javascript
// Validate translation completeness
function validateTranslations() {
  const languages = ['en', 'zh', 'fr', 'es', 'ja'];
  const baseKeys = Object.keys(loadTranslations('en'));
  
  languages.forEach(lang => {
    const translations = loadTranslations(lang);
    const missingKeys = baseKeys.filter(key => !translations[key]);
    
    if (missingKeys.length > 0) {
      console.warn(`Missing translations in ${lang}:`, missingKeys);
    }
  });
}
```

#### Performance Considerations
```javascript
// Cache translations for better performance
const translationCache = new Map();

function getCachedTranslation(key, language) {
  const cacheKey = `${language}:${key}`;
  
  if (translationCache.has(cacheKey)) {
    return translationCache.get(cacheKey);
  }
  
  const translation = getTranslation(key, language);
  translationCache.set(cacheKey, translation);
  
  return translation;
}
```

## ComfyUI V3 Migration - Modern Node Development

### Purpose
ComfyUI V3 schema introduces a more organized way of defining nodes with better type safety, cleaner architecture, and future-proofing. This guide helps migrate existing V1 nodes to the new V3 schema.

### Core Concepts

#### Versioned API
```python
# Use latest ComfyUI API
from comfy_api.latest import ComfyExtension, io, ui

# Use a specific version of ComfyUI API
from comfy_api.v0_0_2 import ComfyExtension, io, ui
```

**Features**:
- **Versioned API**: Future revisions are backwards compatible
- **Latest API**: `comfy_api.latest` points to the latest numbered API
- **Stable API**: Previous version is considered stable
- **Development API**: Current version allows changes without warning

### V1 vs V3 Architecture

#### Key Changes
- **Inputs/Outputs**: Defined by objects instead of dictionaries
- **Execution method**: Fixed to `execute` and is a class method
- **Entry point**: `comfy_entrypoint()` function returns `ComfyExtension` object
- **Node state**: No `__init__` method needed, all methods are class methods
- **Node sanitization**: Node class is sanitized before execution

#### V1 (Legacy)
```python
class MyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {...}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "my_category"

    def execute(self, ...):
        return (result,)

NODE_CLASS_MAPPINGS = {"MyNode": MyNode}
```

#### V3 (Modern)
```python
from comfy_api.latest import ComfyExtension, io

class MyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MyNode",
            display_name="My Node",
            category="my_category",
            inputs=[...],
            outputs=[...]
        )

    @classmethod
    def execute(cls, ...) -> io.NodeOutput:
        return io.NodeOutput(result)

class MyExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MyNode]

async def comfy_entrypoint() -> ComfyExtension:
    return MyExtension()
```

### Migration Steps

#### Step 1: Change Base Class
```python
# V1
class Example:
    def __init__(self):
        pass

# V3
from comfy_api.latest import io

class Example(io.ComfyNode):
    # No __init__ needed
```

#### Step 2: Convert INPUT_TYPES to define_schema
```python
# V1
@classmethod
def INPUT_TYPES(s):
    return {
        "required": {
            "image": ("IMAGE",),
            "int_field": ("INT", {
                "default": 0,
                "min": 0,
                "max": 4096,
                "step": 64,
                "display": "number"
            }),
            "string_field": ("STRING", {
                "multiline": False,
                "default": "Hello"
            }),
            "custom_field": ("MY_CUSTOM_TYPE",),
        },
        "optional": {
            "mask": ("MASK",)
        }
    }

# V3
@classmethod
def define_schema(cls) -> io.Schema:
    return io.Schema(
        node_id="Example",
        display_name="Example Node",
        category="examples",
        description="Node description here",
        inputs=[
            io.Image.Input("image"),
            io.Int.Input("int_field",
                default=0,
                min=0,
                max=4096,
                step=64,
                display_mode=io.NumberDisplay.number
            ),
            io.String.Input("string_field",
                default="Hello",
                multiline=False
            ),
            io.Custom("my_custom_type").Input("custom_input"),
            io.Mask.Input("mask", optional=True)
        ],
        outputs=[
            io.Image.Output()
        ]
    )
```

#### Step 3: Update Execute Method
```python
# V1
def test(self, image, string_field, int_field):
    # Process
    image = 1.0 - image
    return (image,)

# V3
@classmethod
def execute(cls, image, string_field, int_field) -> io.NodeOutput:
    # Process
    image = 1.0 - image

    # Return with optional UI preview
    return io.NodeOutput(image, ui=ui.PreviewImage(image, cls=cls))
```

#### Step 4: Convert Node Properties
| V1 Property | V3 Schema Field | Notes |
|-------------|-----------------|-------|
| RETURN_TYPES | outputs in Schema | List of Output objects |
| RETURN_NAMES | display_name in Output | Per-output display names |
| FUNCTION | Always execute | Method name is standardized |
| CATEGORY | category in Schema | String value |
| OUTPUT_NODE | is_output_node in Schema | Boolean flag |
| DEPRECATED | is_deprecated in Schema | Boolean flag |
| EXPERIMENTAL | is_experimental in Schema | Boolean flag |

#### Step 5: Handle Special Methods

##### Validation
```python
# V1
@classmethod
def VALIDATE_INPUTS(s, **kwargs):
    # Validation logic
    return True

# V3
@classmethod
def validate_inputs(cls, **kwargs) -> bool | str:
    # Return True if valid, error string if not
    if error_condition:
        return "Error message"
    return True
```

##### Lazy Evaluation
```python
# V1
def check_lazy_status(self, image, string_field, ...):
    if condition:
        return ["string_field"]
    return []

# V3
@classmethod
def check_lazy_status(cls, image, string_field, ...):
    if condition:
        return ["string_field"]
    return []
```

##### Cache Control
```python
# V1
@classmethod
def IS_CHANGED(s, **kwargs):
    return "unique_value"

# V3
@classmethod
def fingerprint_inputs(cls, **kwargs):
    return "unique_value"
```

#### Step 6: Create Extension and Entry Point
```python
# V1
NODE_CLASS_MAPPINGS = {
    "Example": Example
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Example Node"
}

# V3
from comfy_api.latest import ComfyExtension

class MyExtension(ComfyExtension):
    # must be declared as async
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Example,
            # Add more nodes here
        ]

# can be declared async or not, both will work
async def comfy_entrypoint() -> MyExtension:
    return MyExtension()
```

### Input Type Reference

#### Basic Types
| V1 Type | V3 Type | Example |
|---------|---------|---------|
| "INT" | io.Int.Input() | io.Int.Input("count", default=1, min=0, max=100) |
| "FLOAT" | io.Float.Input() | io.Float.Input("strength", default=1.0, min=0.0, max=10.0) |
| "STRING" | io.String.Input() | io.String.Input("text", multiline=True) |
| "BOOLEAN" | io.Boolean.Input() | io.Boolean.Input("enabled", default=True) |

#### ComfyUI Types
| V1 Type | V3 Type | Example |
|---------|---------|---------|
| "IMAGE" | io.Image.Input() | io.Image.Input("image", tooltip="Input image") |
| "MASK" | io.Mask.Input() | io.Mask.Input("mask", optional=True) |
| "LATENT" | io.Latent.Input() | io.Latent.Input("latent") |
| "CONDITIONING" | io.Conditioning.Input() | io.Conditioning.Input("positive") |
| "MODEL" | io.Model.Input() | io.Model.Input("model") |
| "VAE" | io.VAE.Input() | io.VAE.Input("vae") |
| "CLIP" | io.CLIP.Input() | io.CLIP.Input("clip") |

#### Combo (Dropdowns)
```python
# V1
"mode": (["option1", "option2", "option3"],)

# V3
io.Combo.Input("mode", options=["option1", "option2", "option3"])
```

### Advanced Features

#### UI Integration
```python
from comfy_api.latest import ui

@classmethod
def execute(cls, images) -> io.NodeOutput:
    # Show preview in node
    return io.NodeOutput(images, ui=ui.PreviewImage(images, cls=cls))
```

#### Output Nodes
```python
@classmethod
def define_schema(cls) -> io.Schema:
    return io.Schema(
        node_id="SaveNode",
        inputs=[...],
        outputs=[],  # Does not need to be empty
        is_output_node=True  # Mark as output node
    )
```

#### Custom Types
```python
from comfy_api.latest import io

# Method 1: Using decorator with class
@io.comfytype(io_type="MY_CUSTOM_TYPE")
class MyCustomType:
    Type = torch.Tensor  # Python type hint

    class Input(io.Input):
        def __init__(self, id: str, **kwargs):
            super().__init__(id, **kwargs)

    class Output(io.Output):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

# Method 2: Using Custom helper
MyCustomType = io.Custom("MY_CUSTOM_TYPE")
```

### Best Practices

#### Type Safety
```python
# Good: Use proper type hints
@classmethod
def execute(cls, image: torch.Tensor, strength: float) -> io.NodeOutput:
    return io.NodeOutput(processed_image)

# Bad: No type hints
@classmethod
def execute(cls, image, strength):
    return io.NodeOutput(processed_image)
```

#### Error Handling
```python
# Good: Proper error handling
@classmethod
def execute(cls, image, strength) -> io.NodeOutput:
    try:
        result = process_image(image, strength)
        return io.NodeOutput(result)
    except Exception as e:
        return io.NodeOutput(f"Error: {str(e)}")
```

#### Documentation
```python
# Good: Comprehensive documentation
@classmethod
def define_schema(cls) -> io.Schema:
    return io.Schema(
        node_id="ImageProcessor",
        display_name="Image Processor",
        category="image_processing",
        description="Processes images with various filters and effects",
        inputs=[
            io.Image.Input("image", tooltip="Input image to process"),
            io.Float.Input("strength", default=1.0, min=0.0, max=2.0, 
                          tooltip="Processing strength multiplier")
        ],
        outputs=[
            io.Image.Output(tooltip="Processed image result")
        ]
    )
```

## ComfyUI Help Page - Rich Node Documentation

### Purpose
ComfyUI's Help Page system allows custom nodes to include rich markdown documentation that will be displayed in the UI instead of generic node descriptions. This provides users with detailed information about node functionality, parameters, and usage examples.

### Features
- **Rich markdown documentation**: Detailed node information in the UI
- **Multi-language support**: Localized documentation for different languages
- **Automatic fallback**: Falls back to default documentation if localized version unavailable
- **Media support**: Images and videos in documentation
- **Tooltip integration**: Basic parameter information from node definitions

### Setup

#### Directory Structure
Create a `docs` folder inside your `WEB_DIRECTORY` with the following structure:

```
my-custom-node/
├── __init__.py
├── web/              # WEB_DIRECTORY
│   ├── js/
│   │   └── my-node.js
│   └── docs/
│       ├── MyNode.md           # Fallback documentation
│       └── MyNode/
│           ├── en.md           # English version
│           ├── zh.md           # Chinese version
│           ├── fr.md           # French version
│           └── de.md           # German version
```

#### File Naming Convention
- **Default documentation**: `NodeName.md` (fallback)
- **Localized documentation**: `NodeName/locale.md` (e.g., `en.md`, `zh.md`, `fr.md`)
- **Node names**: Use dictionary keys from `NODE_CLASS_MAPPINGS`

### Supported Markdown Features

#### Standard Markdown
```markdown
# Heading 1
## Heading 2
### Heading 3

**Bold text**
*Italic text*
`Code snippets`

- Bullet points
- More items

1. Numbered lists
2. Second item

> Blockquotes for important information

[Links](https://example.com)
```

#### Images
```markdown
![Alt text](image.png)
![Example usage](example.png)
![Node preview](preview.jpg)
```

#### Code Blocks
```markdown
```python
# Python code example
def process_image(image, strength):
    return image * strength
```

```javascript
// JavaScript code example
function updateNode() {
    console.log("Node updated");
}
```
```

#### HTML Media Elements
```markdown
<video controls loop muted>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

**Supported video attributes**:
- `controls`: Show video controls
- `autoplay`: Auto-play video
- `loop`: Loop video
- `muted`: Mute video
- `preload`: Preload video
- `poster`: Poster image

### Documentation Examples

#### Basic Node Documentation
```markdown
# Image Processor Node

This node processes images using advanced algorithms with customizable parameters.

## Overview

The Image Processor node applies various filters and effects to input images. It supports multiple processing modes and provides real-time preview of results.

## Parameters

### Required Parameters

- **image**: Input image to process
  - Type: IMAGE
  - Description: The source image that will be processed
  - Tooltip: Select an image to apply processing effects

- **strength**: Processing strength
  - Type: FLOAT
  - Range: 0.0 - 1.0
  - Default: 0.5
  - Description: Controls the intensity of the processing effect

### Optional Parameters

- **mode**: Processing mode
  - Type: COMBO
  - Options: ["blur", "sharpen", "enhance", "filter"]
  - Default: "enhance"
  - Description: Select the type of processing to apply

## Usage Examples

### Basic Usage
1. Connect an image to the **image** input
2. Set the **strength** parameter (0.0 = no effect, 1.0 = full effect)
3. Choose a **mode** from the dropdown
4. The processed image will appear at the output

### Advanced Usage
For best results:
- Use high-resolution images for better quality
- Adjust strength gradually to find the optimal setting
- Combine with other nodes for complex workflows

## Tips and Tricks

- **Performance**: Lower strength values process faster
- **Quality**: Higher resolution images produce better results
- **Workflow**: Use this node early in your pipeline for best performance

## Troubleshooting

### Common Issues

**Node not processing**:
- Check that an image is connected to the input
- Verify the strength parameter is not zero

**Poor quality results**:
- Try adjusting the strength parameter
- Ensure input image is high resolution
- Check that the mode is appropriate for your image type

### Error Messages

- `"No image connected"`: Connect an image to the input
- `"Invalid strength value"`: Ensure strength is between 0.0 and 1.0
- `"Unsupported image format"`: Use supported image formats (PNG, JPG, etc.)
```

#### Advanced Documentation with Media
```markdown
# Advanced Image Processor

A comprehensive image processing node with multiple algorithms and real-time preview.

## Features

- **Multiple algorithms**: Blur, sharpen, enhance, and custom filters
- **Real-time preview**: See results as you adjust parameters
- **Batch processing**: Process multiple images efficiently
- **Quality control**: Fine-tune processing parameters

## Algorithm Details

### Blur Algorithm
The blur algorithm uses Gaussian blur with configurable radius:

```python
def gaussian_blur(image, radius):
    # Apply Gaussian blur with specified radius
    return cv2.GaussianBlur(image, (radius, radius), 0)
```

### Sharpen Algorithm
The sharpen algorithm enhances image details:

```python
def sharpen_image(image, strength):
    # Apply unsharp mask
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel * strength)
```

## Visual Examples

![Before and after comparison](before_after.png)

<video controls loop muted poster="video_poster.jpg">
  <source src="processing_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Performance Guidelines

| Image Size | Processing Time | Memory Usage |
|------------|----------------|--------------|
| 512x512    | ~50ms         | ~10MB        |
| 1024x1024  | ~200ms        | ~40MB        |
| 2048x2048  | ~800ms        | ~160MB       |

## API Reference

### Input Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| image | IMAGE | - | - | Input image to process |
| strength | FLOAT | 0.0-1.0 | 0.5 | Processing intensity |
| mode | COMBO | - | "enhance" | Processing algorithm |
| quality | INT | 1-100 | 85 | Output quality |

### Output Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| processed_image | IMAGE | Processed image result |
| metadata | STRING | Processing metadata |

## Changelog

### Version 1.2.0
- Added new sharpen algorithm
- Improved performance by 30%
- Fixed memory leak in batch processing

### Version 1.1.0
- Added quality parameter
- Enhanced error handling
- Improved documentation

### Version 1.0.0
- Initial release
- Basic blur and enhance algorithms
- Real-time preview
```

#### Multi-Language Documentation

##### English (en.md)
```markdown
# Image Processor Node

This node processes images using advanced algorithms.

## Parameters

- **image**: Input image to process
- **strength**: Processing strength (0.0 - 1.0)

## Usage

![example usage](example.png)

<video controls loop muted>
  <source src="demo.mp4" type="video/mp4">
</video>
```

##### Chinese (zh.md)
```markdown
# 图像处理器节点

此节点使用高级算法处理图像。

## 参数

- **image**: 要处理的输入图像
- **strength**: 处理强度 (0.0 - 1.0)

## 使用方法

![使用示例](example.png)

<video controls loop muted>
  <source src="demo.mp4" type="video/mp4">
</video>
```

##### French (fr.md)
```markdown
# Nœud Processeur d'Image

Ce nœud traite les images en utilisant des algorithmes avancés.

## Paramètres

- **image**: Image d'entrée à traiter
- **strength**: Intensité de traitement (0.0 - 1.0)

## Utilisation

![exemple d'utilisation](example.png)

<video controls loop muted>
  <source src="demo.mp4" type="video/mp4">
</video>
```

### Best Practices

#### Documentation Structure
```markdown
# Node Name

Brief description of what the node does.

## Overview

Detailed explanation of functionality and use cases.

## Parameters

### Required Parameters
- **param1**: Description with type and range
- **param2**: Description with type and range

### Optional Parameters
- **param3**: Description with type and range
- **param4**: Description with type and range

## Usage Examples

### Basic Usage
Step-by-step instructions for common use cases.

### Advanced Usage
Complex scenarios and advanced techniques.

## Tips and Tricks

- Performance optimization tips
- Quality improvement suggestions
- Workflow integration advice

## Troubleshooting

### Common Issues
- Problem descriptions and solutions
- Error message explanations
- Performance considerations

## API Reference

### Input Parameters
Table with all input parameters, types, ranges, and descriptions.

### Output Parameters
Table with all output parameters and descriptions.

## Changelog

### Version X.X.X
- Feature additions
- Bug fixes
- Performance improvements
```

#### Content Guidelines
```markdown
# Good Documentation Practices

## Use Clear Headings
- Use descriptive headings that explain the content
- Maintain consistent heading hierarchy
- Use proper markdown syntax

## Provide Examples
- Include code examples for complex operations
- Show before/after comparisons
- Provide step-by-step tutorials

## Use Visual Aids
- Include screenshots for UI elements
- Add diagrams for complex workflows
- Use videos for dynamic demonstrations

## Write for Your Audience
- Assume basic ComfyUI knowledge
- Explain technical terms
- Provide context for advanced features

## Keep It Updated
- Update documentation with new features
- Remove outdated information
- Test all examples and links
```

#### Media Guidelines
```markdown
# Media Best Practices

## Images
- Use high-quality screenshots
- Optimize file sizes for web
- Include alt text for accessibility
- Use consistent naming conventions

## Videos
- Keep videos short and focused
- Use clear, readable text overlays
- Include audio narration when helpful
- Provide multiple formats when possible

## File Organization
```
docs/
├── MyNode.md
├── MyNode/
│   ├── en.md
│   ├── zh.md
│   └── fr.md
├── images/
│   ├── example.png
│   ├── before_after.jpg
│   └── workflow_diagram.svg
└── videos/
    ├── demo.mp4
    └── tutorial.mp4
```
```

### Advanced Features

#### Dynamic Content
```markdown
# Node with Dynamic Parameters

This node supports dynamic parameter generation based on input.

## Dynamic Parameters

The following parameters are generated dynamically:

- **algorithm_count**: Number of available algorithms
- **quality_presets**: Available quality presets
- **supported_formats**: Supported input formats

## Conditional Parameters

Some parameters only appear when specific conditions are met:

- **blur_radius**: Only visible when mode is "blur"
- **sharpen_strength**: Only visible when mode is "sharpen"
- **filter_type**: Only visible when mode is "filter"
```

#### Interactive Examples
```markdown
# Interactive Node Documentation

## Live Examples

Try these examples with your own images:

### Example 1: Basic Processing
1. Load an image
2. Set strength to 0.5
3. Choose "enhance" mode
4. Observe the results

### Example 2: Advanced Workflow
1. Start with a high-resolution image
2. Apply multiple processing steps
3. Adjust parameters iteratively
4. Export final result

## Parameter Combinations

| Strength | Mode | Result |
|----------|------|--------|
| 0.1 | blur | Light blur effect |
| 0.5 | enhance | Moderate enhancement |
| 1.0 | sharpen | Strong sharpening |
```

### Integration with Node Definitions

#### Tooltip Integration
```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to process"
                }),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Processing strength (0.0 = no effect, 1.0 = full effect)"
                }),
                "mode": (["blur", "sharpen", "enhance", "filter"], {
                    "default": "enhance",
                    "tooltip": "Processing algorithm to use"
                })
            }
        }
```

#### Documentation Access
```python
# The system automatically loads documentation based on:
# 1. User's locale (e.g., zh.md for Chinese users)
# 2. Fallback to default (MyNode.md)
# 3. Tooltip information from node definition
```

### Important Notes

#### File Naming
```markdown
# File Naming Rules

## Node Names
- Use exact names from NODE_CLASS_MAPPINGS
- Case-sensitive
- No spaces or special characters

## Locale Codes
- Use standard locale codes (en, zh, fr, de, etc.)
- Lowercase only
- No country codes (use zh instead of zh-CN)

## Media Files
- Use descriptive names
- Include version numbers for updates
- Use web-friendly formats (PNG, JPG, MP4)
```

#### Performance Considerations
```markdown
# Performance Guidelines

## File Sizes
- Keep images under 1MB when possible
- Compress videos for web delivery
- Use appropriate formats for content type

## Loading Times
- Optimize images for web display
- Consider lazy loading for large media
- Provide fallbacks for slow connections

## Accessibility
- Include alt text for all images
- Provide captions for videos
- Use high contrast for text overlays
```

## ComfyUI Workflow Templates - Example Workflows

### Purpose
ComfyUI's Workflow Templates system allows custom node developers to provide example workflow files that users can discover and use through the template browser. This helps users get started with your nodes and demonstrates their capabilities.

### Features
- **Template browser integration**: Workflows appear in Workflow/Browse Templates menu
- **Thumbnail support**: JPG images for visual preview
- **Automatic categorization**: Templates grouped by custom node module
- **Static file serving**: ComfyUI serves templates and provides API endpoint
- **Multiple folder names**: Flexible naming conventions supported

### Setup

#### Directory Structure
Create an `example_workflows` folder in your custom node directory:

```
ComfyUI-MyCustomNodeModule/
├── __init__.py
├── my_node.py
├── example_workflows/           # Workflow templates folder
│   ├── My_example_workflow_1.json
│   ├── My_example_workflow_1.jpg    # Thumbnail (optional)
│   ├── My_example_workflow_2.json
│   ├── My_example_workflow_2.jpg    # Thumbnail (optional)
│   ├── Advanced_workflow.json
│   └── Advanced_workflow.jpg        # Thumbnail (optional)
└── web/                        # Optional: WEB_DIRECTORY
    └── js/
        └── my_node.js
```

#### Supported Folder Names
ComfyUI accepts multiple folder names for workflow templates:
- **`example_workflows`** (recommended)
- **`workflow`**
- **`workflows`**
- **`example`**
- **`examples`**

### Workflow Template Files

#### JSON Workflow Files
Workflow templates are standard ComfyUI workflow JSON files:

```json
{
  "last_node_id": 10,
  "last_link_id": 5,
  "nodes": [
    {
      "id": 1,
      "type": "LoadImage",
      "pos": [100, 100],
      "size": {"0": 315, "1": 314},
      "flags": {},
      "order": 0,
      "mode": 0,
      "outputs": [
        {"name": "IMAGE", "type": "IMAGE", "links": [1], "slot_index": 0}
      ],
      "properties": {"Node name for S&R": "LoadImage"},
      "widgets_values": ["example_image.png"]
    },
    {
      "id": 2,
      "type": "MyCustomNode",
      "pos": [500, 100],
      "size": {"0": 315, "1": 314},
      "flags": {},
      "order": 1,
      "mode": 0,
      "inputs": [
        {"name": "image", "type": "IMAGE", "link": 1}
      ],
      "outputs": [
        {"name": "processed_image", "type": "IMAGE", "links": [2], "slot_index": 0}
      ],
      "properties": {"Node name for S&R": "MyCustomNode"},
      "widgets_values": [0.5, "enhance"]
    },
    {
      "id": 3,
      "type": "SaveImage",
      "pos": [900, 100],
      "size": {"0": 315, "1": 314},
      "flags": {},
      "order": 2,
      "mode": 0,
      "inputs": [
        {"name": "images", "type": "IMAGE", "link": 2}
      ],
      "properties": {"Node name for S&R": "SaveImage"},
      "widgets_values": ["processed_output"]
    }
  ],
  "links": [
    [1, 1, 0, 2, 0, "IMAGE"],
    [2, 2, 0, 3, 0, "IMAGE"]
  ],
  "groups": [],
  "config": {},
  "extra": {},
  "version": 0.4
}
```

#### Thumbnail Images
Optional JPG images with the same name as the workflow file:

```
example_workflows/
├── My_example_workflow_1.json
├── My_example_workflow_1.jpg    # 300x200px recommended
├── My_example_workflow_2.json
├── My_example_workflow_2.jpg    # 300x200px recommended
└── Advanced_workflow.json
    # No thumbnail - will show default icon
```

### Template Browser Integration

#### User Experience
1. **Access**: Users go to Workflow/Browse Templates menu
2. **Categories**: Templates grouped by custom node module name
3. **Preview**: Thumbnail images show workflow preview
4. **Loading**: Clicking a template loads it into the current workflow

#### API Endpoint
ComfyUI provides an API endpoint for workflow templates:
- **Endpoint**: `/api/workflow_templates`
- **Method**: GET
- **Response**: Collection of workflow templates with metadata

### Best Practices

#### Workflow Design
```json
{
  "workflow_design_guidelines": {
    "node_positioning": "Arrange nodes in logical flow (left to right)",
    "node_spacing": "Use consistent spacing between nodes",
    "grouping": "Group related nodes together",
    "naming": "Use descriptive node names and titles"
  }
}
```

#### Template Organization
```
example_workflows/
├── 01_Basic_Usage.json           # Simple, beginner-friendly
├── 01_Basic_Usage.jpg
├── 02_Advanced_Features.json     # Complex, feature-rich
├── 02_Advanced_Features.jpg
├── 03_Batch_Processing.json      # Batch operations
├── 03_Batch_Processing.jpg
├── 04_Integration_Example.json   # Integration with other nodes
└── 04_Integration_Example.jpg
```

#### Thumbnail Guidelines
```markdown
# Thumbnail Best Practices

## Image Specifications
- **Format**: JPG (recommended) or PNG
- **Size**: 300x200 pixels (3:2 aspect ratio)
- **Quality**: High quality, clear and readable
- **File size**: Under 100KB for fast loading

## Content Guidelines
- **Show workflow overview**: Capture the entire workflow
- **Highlight key nodes**: Make important nodes visible
- **Use clear labels**: Ensure node names are readable
- **Consistent styling**: Use similar styling across thumbnails

## Design Tips
- **High contrast**: Ensure good visibility
- **Clean layout**: Avoid cluttered appearance
- **Focus on flow**: Show the logical progression
- **Brand consistency**: Use consistent colors and fonts
```

### Advanced Examples

#### Basic Usage Template
```json
{
  "template_info": {
    "name": "Basic Image Processing",
    "description": "Simple example showing basic image processing with MyCustomNode",
    "difficulty": "beginner",
    "estimated_time": "2 minutes",
    "required_nodes": ["LoadImage", "MyCustomNode", "SaveImage"]
  },
  "workflow": {
    "nodes": [
      {
        "id": 1,
        "type": "LoadImage",
        "pos": [100, 100],
        "widgets_values": ["example_input.jpg"]
      },
      {
        "id": 2,
        "type": "MyCustomNode",
        "pos": [500, 100],
        "widgets_values": [0.5, "enhance"]
      },
      {
        "id": 3,
        "type": "SaveImage",
        "pos": [900, 100],
        "widgets_values": ["output"]
      }
    ]
  }
}
```

#### Advanced Features Template
```json
{
  "template_info": {
    "name": "Advanced Image Processing Pipeline",
    "description": "Complex workflow demonstrating advanced features and integration",
    "difficulty": "advanced",
    "estimated_time": "10 minutes",
    "required_nodes": ["LoadImage", "MyCustomNode", "ControlNet", "Upscale", "SaveImage"]
  },
  "workflow": {
    "nodes": [
      {
        "id": 1,
        "type": "LoadImage",
        "pos": [100, 100],
        "widgets_values": ["input_image.jpg"]
      },
      {
        "id": 2,
        "type": "MyCustomNode",
        "pos": [500, 100],
        "widgets_values": [0.8, "sharpen"]
      },
      {
        "id": 3,
        "type": "ControlNet",
        "pos": [500, 300],
        "widgets_values": ["canny", 0.5]
      },
      {
        "id": 4,
        "type": "Upscale",
        "pos": [900, 100],
        "widgets_values": [2.0, "lanczos"]
      },
      {
        "id": 5,
        "type": "SaveImage",
        "pos": [1300, 100],
        "widgets_values": ["final_output"]
      }
    ]
  }
}
```

#### Batch Processing Template
```json
{
  "template_info": {
    "name": "Batch Image Processing",
    "description": "Process multiple images with consistent settings",
    "difficulty": "intermediate",
    "estimated_time": "5 minutes",
    "required_nodes": ["LoadImageBatch", "MyCustomNode", "SaveImageBatch"]
  },
  "workflow": {
    "nodes": [
      {
        "id": 1,
        "type": "LoadImageBatch",
        "pos": [100, 100],
        "widgets_values": ["input_folder/", "*.jpg"]
      },
      {
        "id": 2,
        "type": "MyCustomNode",
        "pos": [500, 100],
        "widgets_values": [0.6, "enhance"]
      },
      {
        "id": 3,
        "type": "SaveImageBatch",
        "pos": [900, 100],
        "widgets_values": ["output_folder/", "processed_"]
      }
    ]
  }
}
```

### Template Metadata

#### Workflow Information
```json
{
  "template_metadata": {
    "title": "My Custom Node Example",
    "description": "This workflow demonstrates the basic usage of MyCustomNode",
    "author": "Your Name",
    "version": "1.0.0",
    "created_date": "2024-01-15",
    "updated_date": "2024-01-20",
    "tags": ["image-processing", "beginner", "example"],
    "difficulty": "beginner",
    "estimated_time": "2 minutes",
    "required_nodes": ["LoadImage", "MyCustomNode", "SaveImage"],
    "prerequisites": ["Basic ComfyUI knowledge", "Image files"],
    "output_description": "Processed image with enhanced details"
  }
}
```

#### Template Categories
```json
{
  "template_categories": {
    "beginner": {
      "description": "Simple workflows for getting started",
      "examples": ["Basic Usage", "Simple Processing"]
    },
    "intermediate": {
      "description": "Workflows with moderate complexity",
      "examples": ["Batch Processing", "Multi-step Pipeline"]
    },
    "advanced": {
      "description": "Complex workflows with multiple integrations",
      "examples": ["Advanced Pipeline", "Custom Integration"]
    }
  }
}
```

### Integration with Documentation

#### Template Documentation
```markdown
# Workflow Templates

## Available Templates

### 1. Basic Usage
- **File**: `01_Basic_Usage.json`
- **Description**: Simple image processing workflow
- **Difficulty**: Beginner
- **Time**: 2 minutes
- **Nodes**: LoadImage → MyCustomNode → SaveImage

### 2. Advanced Features
- **File**: `02_Advanced_Features.json`
- **Description**: Complex workflow with multiple features
- **Difficulty**: Advanced
- **Time**: 10 minutes
- **Nodes**: LoadImage → MyCustomNode → ControlNet → Upscale → SaveImage

### 3. Batch Processing
- **File**: `03_Batch_Processing.json`
- **Description**: Process multiple images efficiently
- **Difficulty**: Intermediate
- **Time**: 5 minutes
- **Nodes**: LoadImageBatch → MyCustomNode → SaveImageBatch
```

#### Template README
```markdown
# Workflow Templates

This directory contains example workflows for MyCustomNode.

## Template Files

- `01_Basic_Usage.json` - Simple workflow for beginners
- `02_Advanced_Features.json` - Complex workflow with advanced features
- `03_Batch_Processing.json` - Batch processing workflow
- `04_Integration_Example.json` - Integration with other nodes

## Thumbnail Images

- `01_Basic_Usage.jpg` - Preview of basic workflow
- `02_Advanced_Features.jpg` - Preview of advanced workflow
- `03_Batch_Processing.jpg` - Preview of batch workflow
- `04_Integration_Example.jpg` - Preview of integration workflow

## Usage

1. Open ComfyUI
2. Go to Workflow → Browse Templates
3. Find "MyCustomNode" category
4. Click on a template to load it
5. Adjust parameters as needed
6. Run the workflow

## Customization

- Modify node parameters to suit your needs
- Add or remove nodes as required
- Save your customized workflow
- Share with the community
```

### Important Notes

#### File Naming
```markdown
# File Naming Guidelines

## Workflow Files
- Use descriptive names: `Basic_Usage.json`, `Advanced_Features.json`
- Avoid spaces: Use underscores or hyphens
- Include version numbers for updates: `Basic_Usage_v2.json`
- Keep names short but descriptive

## Thumbnail Files
- Match workflow file names exactly
- Use JPG format for better compression
- Include descriptive alt text in filenames
- Use consistent naming conventions
```

#### Performance Considerations
```markdown
# Performance Guidelines

## File Sizes
- Keep JSON files under 1MB
- Compress thumbnail images
- Use appropriate image formats
- Optimize for web delivery

## Loading Times
- Minimize workflow complexity for templates
- Use efficient node arrangements
- Avoid unnecessary connections
- Test loading performance

## User Experience
- Provide clear workflow descriptions
- Include helpful comments in workflows
- Use intuitive node names
- Organize templates by difficulty
```

## ComfyUI Registry - Publishing Custom Nodes

### Purpose
The ComfyUI Registry is a public collection of custom nodes that allows developers to publish, version, deprecate, and track metrics related to their custom nodes. Users can discover, install, and rate custom nodes from the registry, creating a standardized ecosystem for ComfyUI extensions.

### Features
- **Node versioning**: Semantic versioning for reliable workflow reproduction
- **Node security**: Security scanning and verification for safe installations
- **Search functionality**: Discover existing nodes for workflows
- **Community ratings**: User feedback and quality metrics
- **Workflow compatibility**: Version tracking in workflow JSON files
- **Standardized development**: Community standards for node development

### Why Use the Registry?

#### Node Versioning
```json
{
  "versioning_benefits": {
    "semantic_versioning": "Major.Minor.Patch versioning system",
    "workflow_reproduction": "Reliable recreation of workflows with specific node versions",
    "safe_upgrades": "Users can choose when to upgrade node versions",
    "version_locking": "Lock specific versions to prevent breaking changes",
    "deprecation_management": "Graceful handling of deprecated node versions"
  }
}
```

#### Node Security
```json
{
  "security_features": {
    "malicious_scanning": "Automatic scanning for malicious behavior",
    "pip_wheel_checks": "Verification of custom pip wheels",
    "system_call_monitoring": "Detection of arbitrary system calls",
    "verification_flags": "Visual indicators for verified nodes",
    "security_standards": "Comprehensive security standards compliance"
  }
}
```

#### Search and Discovery
```json
{
  "discovery_features": {
    "registry_search": "Search across all registered nodes",
    "workflow_integration": "Find nodes suitable for specific workflows",
    "community_ratings": "User feedback and quality metrics",
    "categorization": "Organized node categories and tags",
    "documentation": "Centralized documentation and examples"
  }
}
```

### Publishing Nodes

#### Getting Started
```markdown
# Publishing Your First Node

## Prerequisites
- Complete custom node development
- GitHub repository with your node
- Semantic versioning understanding
- Security compliance verification

## Publishing Process
1. **Prepare your node**: Ensure it follows registry standards
2. **Security review**: Pass security scanning requirements
3. **Version management**: Implement semantic versioning
4. **Documentation**: Provide comprehensive documentation
5. **Registry submission**: Submit to the ComfyUI Registry
6. **Community review**: Community feedback and approval
7. **Publication**: Node becomes available to users
```

#### Registry Standards
```json
{
  "registry_standards": {
    "node_structure": {
      "required_files": ["__init__.py", "node_definition.py"],
      "documentation": "README.md with usage examples",
      "version_info": "Version information in node files",
      "dependencies": "requirements.txt with all dependencies"
    },
    "security_requirements": {
      "no_malicious_code": "No arbitrary system calls",
      "safe_dependencies": "Verified pip packages only",
      "no_network_calls": "No unauthorized network requests",
      "code_review": "Community code review process"
    },
    "versioning_requirements": {
      "semantic_versioning": "Follow semver.org standards",
      "changelog": "Detailed changelog for each version",
      "breaking_changes": "Clear documentation of breaking changes",
      "deprecation_policy": "Graceful deprecation handling"
    }
  }
}
```

### Node Versioning

#### Semantic Versioning
```markdown
# Semantic Versioning for ComfyUI Nodes

## Version Format: MAJOR.MINOR.PATCH
- **MAJOR**: Breaking changes that require workflow updates
- **MINOR**: New features that are backward compatible
- **PATCH**: Bug fixes that are backward compatible

## Examples
- `1.0.0`: Initial release
- `1.0.1`: Bug fix (backward compatible)
- `1.1.0`: New feature (backward compatible)
- `2.0.0`: Breaking change (requires workflow updates)

## Workflow Integration
```json
{
  "workflow_versioning": {
    "node_versions": {
      "MyCustomNode": "1.2.3",
      "AnotherNode": "2.1.0"
    },
    "compatibility": "Workflows store exact node versions",
    "reproduction": "Reliable workflow reproduction",
    "upgrade_path": "Clear upgrade instructions"
  }
}
```

#### Version Management
```python
# Node version definition
class MyCustomNode:
    VERSION = "1.2.3"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0
                })
            }
        }
    
    @classmethod
    def get_version_info(cls):
        return {
            "version": cls.VERSION,
            "compatibility": ">=1.0.0",
            "breaking_changes": [],
            "new_features": ["Enhanced processing", "Better performance"],
            "bug_fixes": ["Fixed memory leak", "Improved error handling"]
        }
```

### Security Standards

#### Security Scanning
```json
{
  "security_scanning": {
    "malicious_code_detection": {
      "system_calls": "Scan for arbitrary system calls",
      "network_requests": "Detect unauthorized network requests",
      "file_operations": "Monitor file system access",
      "process_execution": "Check for process spawning"
    },
    "dependency_verification": {
      "pip_packages": "Verify all pip dependencies",
      "custom_wheels": "Scan custom wheel files",
      "version_pinning": "Ensure dependency version pinning",
      "vulnerability_scanning": "Check for known vulnerabilities"
    },
    "code_analysis": {
      "static_analysis": "Automated code analysis",
      "dynamic_analysis": "Runtime behavior monitoring",
      "pattern_detection": "Detect suspicious code patterns",
      "community_review": "Human code review process"
    }
  }
}
```

#### Security Best Practices
```python
# Security-compliant node example
import os
import json
from pathlib import Path

class SecureCustomNode:
    """
    Example of a security-compliant custom node
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("STRING", {"multiline": True})
            }
        }
    
    @classmethod
    def execute(cls, input_data):
        # Safe operations only
        try:
            # No arbitrary system calls
            # No network requests
            # No file system access outside allowed directories
            result = cls.process_data_safely(input_data)
            return (result,)
        except Exception as e:
            # Proper error handling
            return (f"Error: {str(e)}",)
    
    @classmethod
    def process_data_safely(cls, data):
        # Only safe data processing
        # No external dependencies
        # No system calls
        return data.upper()
```

### Registry Integration

#### Node Registration
```python
# Registry-compliant node structure
class RegistryCompliantNode:
    """
    A node that follows registry standards
    """
    
    # Version information
    VERSION = "1.0.0"
    AUTHOR = "Your Name"
    DESCRIPTION = "A registry-compliant custom node"
    CATEGORY = "Custom/Registry"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {"default": "Hello World"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    
    @classmethod
    def get_registry_info(cls):
        return {
            "version": cls.VERSION,
            "author": cls.AUTHOR,
            "description": cls.DESCRIPTION,
            "category": cls.CATEGORY,
            "security_verified": True,
            "dependencies": [],
            "compatibility": {
                "comfyui_version": ">=0.0.1",
                "python_version": ">=3.8"
            }
        }
    
    def process(self, input):
        return (input,)
```

#### Registry Metadata
```json
{
  "registry_metadata": {
    "node_id": "comfyui-registry-compliant-node",
    "name": "Registry Compliant Node",
    "description": "A node that follows registry standards",
    "version": "1.0.0",
    "author": "Your Name",
    "license": "MIT",
    "repository": "https://github.com/yourusername/your-node",
    "documentation": "https://github.com/yourusername/your-node#readme",
    "tags": ["example", "registry", "compliant"],
    "category": "Custom/Registry",
    "security_verified": true,
    "dependencies": [],
    "compatibility": {
      "comfyui_version": ">=0.0.1",
      "python_version": ">=3.8"
    },
    "metrics": {
      "downloads": 0,
      "rating": 0,
      "reviews": 0
    }
  }
}
```

### Community Features

#### User Ratings and Reviews
```json
{
  "community_features": {
    "user_ratings": {
      "rating_system": "1-5 star rating system",
      "review_comments": "Detailed user reviews",
      "helpfulness_voting": "Vote on review helpfulness",
      "rating_aggregation": "Average rating calculation"
    },
    "discovery": {
      "search_functionality": "Search by name, description, tags",
      "category_browsing": "Browse by node categories",
      "trending_nodes": "Popular and trending nodes",
      "recommendations": "Personalized node recommendations"
    },
    "workflow_integration": {
      "workflow_templates": "Example workflows using the node",
      "compatibility_info": "Compatibility with other nodes",
      "version_tracking": "Track node versions in workflows",
      "upgrade_notifications": "Notify users of available updates"
    }
  }
}
```

#### Quality Metrics
```json
{
  "quality_metrics": {
    "download_statistics": {
      "total_downloads": "Total number of downloads",
      "monthly_downloads": "Downloads per month",
      "version_distribution": "Download distribution by version",
      "geographic_distribution": "Downloads by geographic region"
    },
    "user_engagement": {
      "rating_distribution": "Distribution of user ratings",
      "review_count": "Number of user reviews",
      "workflow_usage": "Usage in community workflows",
      "issue_reports": "Bug reports and feature requests"
    },
    "technical_metrics": {
      "compatibility_score": "Compatibility with ComfyUI versions",
      "performance_metrics": "Node execution performance",
      "error_rates": "Error rates in user workflows",
      "update_frequency": "Frequency of node updates"
    }
  }
}
```

### Best Practices

#### Registry Compliance
```markdown
# Registry Compliance Checklist

## Development Standards
- [ ] Follow semantic versioning
- [ ] Implement proper error handling
- [ ] Use secure coding practices
- [ ] Provide comprehensive documentation
- [ ] Include example workflows
- [ ] Test with multiple ComfyUI versions

## Security Requirements
- [ ] No arbitrary system calls
- [ ] No unauthorized network requests
- [ ] Safe dependency management
- [ ] Code review by community
- [ ] Security scanning compliance
- [ ] Regular security updates

## Documentation Standards
- [ ] Clear README with examples
- [ ] API documentation
- [ ] Changelog for each version
- [ ] Installation instructions
- [ ] Troubleshooting guide
- [ ] Community support channels
```

#### Version Management
```python
# Version management best practices
class VersionedNode:
    VERSION = "1.2.3"
    
    @classmethod
    def get_version_info(cls):
        return {
            "version": cls.VERSION,
            "changelog": {
                "1.2.3": {
                    "type": "patch",
                    "changes": ["Fixed memory leak", "Improved performance"],
                    "breaking_changes": False
                },
                "1.2.0": {
                    "type": "minor",
                    "changes": ["Added new feature", "Enhanced UI"],
                    "breaking_changes": False
                },
                "1.0.0": {
                    "type": "major",
                    "changes": ["Initial release"],
                    "breaking_changes": False
                }
            },
            "migration_guide": {
                "1.2.3": "No migration required",
                "1.2.0": "No migration required"
            }
        }
```

### Important Notes

#### Registry Benefits
```markdown
# Registry Benefits for Developers

## Standardization
- Consistent development practices
- Community-approved standards
- Quality assurance processes
- Security verification

## Discovery
- Increased node visibility
- Community ratings and reviews
- Search functionality
- Workflow integration

## Maintenance
- Version management
- Dependency tracking
- Security updates
- Community support
```

#### User Benefits
```markdown
# Registry Benefits for Users

## Safety
- Security-verified nodes
- Malicious code protection
- Safe dependency management
- Community oversight

## Reliability
- Version tracking in workflows
- Reproducible workflows
- Upgrade notifications
- Compatibility information

## Discovery
- Search functionality
- Community ratings
- Example workflows
- Documentation
```

## ComfyUI Registry Publishing - Step-by-Step Guide

### Purpose
This guide provides step-by-step instructions for publishing custom nodes to the ComfyUI Registry, including account setup, metadata configuration, and automated publishing workflows.

### Features
- **Account setup**: Create publisher accounts and API keys
- **Metadata management**: Configure node metadata in pyproject.toml
- **Manual publishing**: Use Comfy CLI for manual publishing
- **Automated publishing**: GitHub Actions for automated publishing
- **Version management**: Semantic versioning and automated updates

### Step 1: Set Up a Registry Account

#### Create a Publisher
```markdown
# Creating a Publisher Account

## Steps
1. **Go to Comfy Registry**: Visit the registry website
2. **Create publisher account**: Set up your publisher identity
3. **Choose publisher ID**: Select a globally unique identifier
4. **Profile setup**: Complete your publisher profile

## Important Notes
- **Publisher ID**: Globally unique, cannot be changed later
- **URL format**: Used in custom node URLs
- **Profile page**: Publisher ID found after @ symbol
```

#### Create an API Key
```markdown
# Creating an API Key

## Steps
1. **Navigate to API keys**: Go to publisher settings
2. **Select publisher**: Choose the publisher for the API key
3. **Name the key**: Give your API key a descriptive name
4. **Save securely**: Store the key safely (cannot be recovered)

## Security Notes
- **Safe storage**: Save API key in secure location
- **No recovery**: Lost keys cannot be recovered
- **Access control**: Limit key access to necessary personnel
```

### Step 2: Add Metadata

#### Install Comfy CLI
```bash
# Install Comfy CLI
pip install comfy-cli

# Verify installation
comfy --version
```

#### Initialize Node Metadata
```bash
# Initialize node metadata
comfy node init
```

#### Generated pyproject.toml
```toml
# pyproject.toml
[project]
name = "" # Unique identifier for your node. Immutable after creation.
description = ""
version = "1.0.0" # Custom Node version. Must be semantically versioned.
license = { file = "LICENSE.txt" }
dependencies = [] # Filled in from requirements.txt

[project.urls]
Repository = "https://github.com/..."

[tool.comfy]
PublisherId = "" # TODO (fill in Publisher ID from Comfy Registry Website).
DisplayName = "" # Display name for the Custom Node. Can be changed later.
Icon = "https://example.com/icon.png" # SVG, PNG, JPG or GIF (MAX. 800x400px)
```

#### Complete Metadata Configuration
```toml
# Complete pyproject.toml example
[project]
name = "my-awesome-node"
description = "An awesome custom node for ComfyUI"
version = "1.0.0"
license = { file = "LICENSE.txt" }
dependencies = [
    "torch>=1.9.0",
    "pillow>=8.3.0",
    "numpy>=1.21.0"
]

[project.urls]
Repository = "https://github.com/yourusername/ComfyUI-MyAwesomeNode"
Documentation = "https://github.com/yourusername/ComfyUI-MyAwesomeNode#readme"
Issues = "https://github.com/yourusername/ComfyUI-MyAwesomeNode/issues"

[tool.comfy]
PublisherId = "your-publisher-id"
DisplayName = "My Awesome Node"
Icon = "https://raw.githubusercontent.com/yourusername/ComfyUI-MyAwesomeNode/main/icon.png"
Category = "Custom/Image Processing"
Tags = ["image", "processing", "awesome"]
```

### Step 3: Publish to the Registry

#### Option 1: Manual Publishing with Comfy CLI
```bash
# Publish node manually
comfy node publish

# You'll be prompted for the API key
# API Key for publisher '<publisher id>': ****************************************************
# ...Version 1.0.0 Published.
# See it here: https://registry.comfy.org/publisherId/your-node
```

#### API Key Handling
```markdown
# API Key Best Practices

## Copy-Paste Issues
- **Windows CTRL+V**: May add \x16 at the end
- **Right-click paste**: Recommended for clean pasting
- **Manual typing**: Alternative for short keys

## Security
- **Hidden input**: API key is hidden by default
- **Secure storage**: Store in password manager
- **Access control**: Limit who has access to keys
```

#### Option 2: Automated Publishing with GitHub Actions

##### Set Up GitHub Secret
```markdown
# GitHub Secret Setup

## Steps
1. **Go to repository settings**: Settings -> Secrets and Variables -> Actions
2. **Create new secret**: Under Repository secrets -> New Repository Secret
3. **Name the secret**: `REGISTRY_ACCESS_TOKEN`
4. **Add API key**: Store your API key as the value
5. **Save securely**: Confirm the secret is saved
```

##### Create GitHub Action
```yaml
# .github/workflows/publish_action.yml
name: Publish to Comfy registry
on:
  workflow_dispatch:
  push:
    branches:
      - main
    paths:
      - "pyproject.toml"

jobs:
  publish-node:
    name: Publish Custom Node to registry
    runs-on: ubuntu-latest
    steps:
      - name: Check out code
        uses: actions/checkout@v4
      - name: Publish Custom Node
        uses: Comfy-Org/publish-node-action@main
        with:
          personal_access_token: ${{ secrets.REGISTRY_ACCESS_TOKEN }}
```

##### Customize for Different Branches
```yaml
# For different branch names (e.g., master)
on:
  workflow_dispatch:
  push:
    branches:
      - master  # Change from 'main' to 'master'
    paths:
      - "pyproject.toml"
```

##### Test the GitHub Action
```markdown
# Testing Automated Publishing

## Steps
1. **Update version**: Change version in pyproject.toml
2. **Commit changes**: Commit and push to repository
3. **Check action**: Monitor GitHub Actions tab
4. **Verify registry**: Check registry for updated node
5. **Test workflow**: Ensure action runs successfully

## Triggers
- **Manual trigger**: workflow_dispatch
- **Automatic trigger**: Push to main branch with pyproject.toml changes
- **Version updates**: Any change to pyproject.toml triggers publish
```

### Advanced Publishing Features

#### Version Management
```toml
# Semantic versioning in pyproject.toml
[project]
version = "1.2.3"  # MAJOR.MINOR.PATCH

# Version bumping examples
# 1.0.0 -> 1.0.1  # Patch (bug fixes)
# 1.0.1 -> 1.1.0  # Minor (new features)
# 1.1.0 -> 2.0.0  # Major (breaking changes)
```

#### Dependency Management
```toml
# Dependencies from requirements.txt
[project]
dependencies = [
    "torch>=1.9.0",
    "pillow>=8.3.0",
    "numpy>=1.21.0",
    "opencv-python>=4.5.0"
]

# Optional dependencies
[project.optional-dependencies]
dev = [
    "pytest>=6.0.0",
    "black>=21.0.0",
    "flake8>=3.8.0"
]
```

#### Metadata Validation
```markdown
# Metadata Requirements

## Required Fields
- **name**: Unique identifier (immutable)
- **description**: Clear description of functionality
- **version**: Semantic version (e.g., 1.0.0)
- **PublisherId**: Your publisher ID from registry
- **DisplayName**: User-friendly display name

## Optional Fields
- **Icon**: URL to icon image (max 800x400px)
- **Category**: Node category for organization
- **Tags**: Searchable tags for discovery
- **Repository**: GitHub repository URL
- **Documentation**: Documentation URL
- **Issues**: Issues tracker URL
```

### Best Practices

#### Publishing Workflow
```markdown
# Publishing Best Practices

## Pre-Publishing Checklist
- [ ] Test node functionality thoroughly
- [ ] Update version in pyproject.toml
- [ ] Verify all metadata is correct
- [ ] Test with multiple ComfyUI versions
- [ ] Update documentation if needed
- [ ] Create/update example workflows
- [ ] Review security compliance

## Publishing Process
1. **Development**: Complete node development
2. **Testing**: Test thoroughly in development
3. **Version bump**: Update version number
4. **Commit**: Commit changes to repository
5. **Push**: Push to main branch
6. **Verify**: Check registry for publication
7. **Test**: Test published node installation
```

#### Security Considerations
```markdown
# Security Best Practices

## API Key Security
- **Secure storage**: Use password manager
- **Access control**: Limit key access
- **Rotation**: Rotate keys periodically
- **Monitoring**: Monitor key usage

## Code Security
- **No secrets**: Don't commit API keys
- **Dependency scanning**: Scan for vulnerabilities
- **Code review**: Review all code changes
- **Security testing**: Test for security issues
```

#### Version Management
```markdown
# Version Management Best Practices

## Semantic Versioning
- **MAJOR**: Breaking changes (2.0.0)
- **MINOR**: New features (1.1.0)
- **PATCH**: Bug fixes (1.0.1)

## Version Bumping
- **Bug fixes**: Increment patch version
- **New features**: Increment minor version
- **Breaking changes**: Increment major version
- **Pre-release**: Use pre-release versions (1.0.0-alpha.1)
```

### Troubleshooting

#### Common Issues
```markdown
# Common Publishing Issues

## API Key Issues
- **Invalid key**: Verify key is correct
- **Expired key**: Create new key if expired
- **Permission issues**: Check key permissions
- **Copy-paste issues**: Use right-click paste

## Metadata Issues
- **Missing fields**: Ensure all required fields are present
- **Invalid format**: Check TOML syntax
- **Version format**: Use semantic versioning
- **Publisher ID**: Verify publisher ID is correct

## GitHub Actions Issues
- **Secret not found**: Check secret name and value
- **Permission denied**: Verify repository permissions
- **Workflow not triggered**: Check branch and path filters
- **Action failed**: Check action logs for errors
```

#### Debugging Steps
```bash
# Debug publishing issues
comfy node publish --verbose

# Check node metadata
comfy node info

# Validate pyproject.toml
comfy node validate

# Test without publishing
comfy node publish --dry-run
```

### Important Notes

#### Registry Benefits
```markdown
# Registry Benefits

## For Developers
- **Automated publishing**: GitHub Actions integration
- **Version management**: Semantic versioning
- **Security scanning**: Automatic security checks
- **Community feedback**: User ratings and reviews

## For Users
- **Safe installation**: Security-verified nodes
- **Version tracking**: Reliable workflow reproduction
- **Easy discovery**: Search and categorization
- **Quality assurance**: Community oversight
```

#### Publishing Timeline
```markdown
# Publishing Timeline

## Initial Publication
1. **Setup**: 5-10 minutes
2. **Metadata**: 10-15 minutes
3. **First publish**: 2-3 minutes
4. **Verification**: 5-10 minutes

## Updates
1. **Version bump**: 1-2 minutes
2. **Commit/push**: 1-2 minutes
3. **Automated publish**: 2-3 minutes
4. **Registry update**: 1-2 minutes
```

## ComfyUI Registry - Claim My Node Feature

### Purpose
The "Claim My Node" feature allows developers to claim ownership of custom nodes in the ComfyUI Registry that were migrated from the legacy ComfyUI Manager system. This ensures proper ownership, security, and accountability within the community.

### Features
- **Node ownership**: Claim migrated nodes from legacy system
- **GitHub authentication**: Verify repository admin access
- **Publisher management**: Create and manage publishers
- **Security verification**: Ensure rightful ownership
- **Smooth migration**: Transition from legacy to new registry standards

### What are Unclaimed Nodes?

#### Legacy System Migration
```markdown
# Unclaimed Nodes Overview

## What are Unclaimed Nodes?
- **Legacy migration**: Nodes from ComfyUI Manager legacy system
- **New standards**: Migrated to Comfy Registry with latest standards
- **Waiting for claim**: Original authors need to claim ownership
- **Smooth transition**: Maintain proper ownership and control

## Migration Process
1. **Legacy system**: Originally published in ComfyUI Manager
2. **Registry migration**: Moved to Comfy Registry with new standards
3. **Unclaimed status**: Waiting for original authors to claim
4. **Ownership verification**: Ensure proper ownership and control
```

#### Node Status
```json
{
  "node_status": {
    "unclaimed": {
      "description": "Nodes migrated from legacy system",
      "status": "Waiting for original author to claim",
      "action_required": "Claim ownership through registry",
      "benefits": "Full control and management capabilities"
    },
    "claimed": {
      "description": "Nodes with verified ownership",
      "status": "Fully managed by original author",
      "capabilities": "Update, version, and manage node",
      "security": "Verified ownership and permissions"
    }
  }
}
```

### Getting Started

#### Step 1: Navigate to Unclaimed Node Page
```markdown
# Accessing Unclaimed Nodes

## Steps
1. **Visit registry**: Go to Comfy Registry website
2. **Find unclaimed nodes**: Look for "unclaimed" status
3. **Click claim button**: "Claim my node!" button
4. **Review node details**: Check node information and requirements

## Node Information
- **Node name**: Original node name
- **Repository link**: GitHub repository URL
- **Publisher status**: Current ownership status
- **Migration details**: Legacy system information
```

#### Step 2: Create a Publisher (if needed)
```markdown
# Publisher Creation

## When Required
- **No existing publisher**: First time using registry
- **New publisher needed**: Want to use different publisher
- **Account setup**: Complete publisher profile

## Publisher Creation Process
1. **Account setup**: Create publisher account
2. **Choose publisher ID**: Globally unique identifier
3. **Profile completion**: Complete publisher profile
4. **Verification**: Verify publisher information
```

#### Step 3: Select Publisher
```markdown
# Publisher Selection

## Selection Process
1. **Choose publisher**: Select from existing publishers
2. **Verify permissions**: Ensure admin access
3. **Redirect to claim**: Navigate to claim page
4. **Review details**: Check node and publisher information

## Publisher Requirements
- **Admin access**: Must have admin rights to repository
- **GitHub account**: Logged in to correct GitHub account
- **Repository access**: Admin privileges for node repository
```

### Claim Process

#### Step 1: Review Node Information
```markdown
# Node Information Review

## Information to Check
- **Node name**: Verify correct node
- **Repository link**: Check GitHub repository
- **Publisher status**: Current ownership status
- **Migration details**: Legacy system information

## Verification Steps
1. **Node details**: Confirm node information
2. **Repository access**: Verify GitHub repository
3. **Admin privileges**: Check admin access
4. **Publisher selection**: Confirm publisher choice
```

#### Step 2: GitHub Authentication
```markdown
# GitHub Authentication Process

## Authentication Steps
1. **Click "Continue with GitHub"**: Start authentication
2. **Login verification**: Ensure correct GitHub account
3. **Admin rights check**: Verify admin privileges
4. **Repository access**: Confirm repository permissions

## Requirements
- **GitHub account**: Must be logged in to correct account
- **Admin rights**: Repository admin privileges required
- **Repository access**: Full access to node repository
- **Authentication**: Complete GitHub OAuth process
```

#### Step 3: Verify Admin Access
```markdown
# Admin Access Verification

## Verification Process
1. **Repository check**: Verify GitHub repository access
2. **Admin privileges**: Confirm admin rights
3. **Permission validation**: Check repository permissions
4. **Access confirmation**: Ensure full repository access

## Requirements
- **Admin privileges**: Repository admin rights
- **Repository access**: Full access to node repository
- **Permission verification**: Confirm admin capabilities
- **Access validation**: Verify repository permissions
```

#### Step 4: Claim the Node
```markdown
# Node Claiming Process

## Claiming Steps
1. **Click "Claim"**: Initiate claiming process
2. **Verification**: Confirm admin access
3. **Ownership transfer**: Change publisher status
4. **Completion**: Node ownership confirmed

## Result
- **Publisher status**: Changes to show ownership
- **Node management**: Full control over node
- **Update capabilities**: Can update and manage node
- **Security**: Verified ownership and permissions
```

#### Step 5: Complete!
```markdown
# Claim Completion

## What Happens Next
1. **Ownership confirmed**: Node ownership verified
2. **Management access**: Full node management capabilities
3. **Update permissions**: Can update and version node
4. **Registry integration**: Node fully integrated with registry

## Next Steps
- **Node management**: Update and maintain node
- **Version control**: Manage node versions
- **Registry publishing**: Publish updates to registry
- **Community engagement**: Manage user feedback
```

### Advanced Features

#### Publisher Management
```markdown
# Publisher Management

## Publisher Capabilities
- **Node ownership**: Manage multiple nodes
- **Version control**: Update and version nodes
- **Registry publishing**: Publish to registry
- **Community management**: Handle user feedback

## Publisher Requirements
- **Admin access**: Repository admin privileges
- **GitHub account**: Verified GitHub account
- **Repository ownership**: Full repository access
- **Registry integration**: Registry account setup
```

#### Node Management
```markdown
# Node Management After Claiming

## Management Capabilities
- **Node updates**: Update node functionality
- **Version control**: Manage node versions
- **Registry publishing**: Publish to registry
- **Community engagement**: Handle user feedback

## Security Features
- **Ownership verification**: Verified node ownership
- **Admin access**: Repository admin privileges
- **Registry integration**: Full registry capabilities
- **Community oversight**: Community feedback and ratings
```

### Best Practices

#### Claiming Process
```markdown
# Claiming Best Practices

## Pre-Claiming Checklist
- [ ] Verify GitHub account access
- [ ] Confirm repository admin rights
- [ ] Check node information accuracy
- [ ] Review publisher requirements
- [ ] Prepare for claim process

## Claiming Process
1. **Account verification**: Ensure correct GitHub account
2. **Repository access**: Verify admin privileges
3. **Node review**: Check node information
4. **Publisher setup**: Create or select publisher
5. **Claim initiation**: Start claiming process
6. **Verification**: Complete admin access verification
7. **Completion**: Confirm node ownership
```

#### Security Considerations
```markdown
# Security Best Practices

## Account Security
- **GitHub account**: Use secure GitHub account
- **Repository access**: Limit admin access to necessary personnel
- **Authentication**: Use strong authentication methods
- **Access control**: Monitor repository access

## Registry Security
- **Publisher verification**: Verify publisher identity
- **Node ownership**: Confirm rightful ownership
- **Registry integration**: Secure registry access
- **Community oversight**: Community feedback and ratings
```

### Troubleshooting

#### Common Issues
```markdown
# Common Claiming Issues

## Authentication Issues
- **GitHub login**: Ensure correct GitHub account
- **Admin access**: Verify repository admin rights
- **Repository access**: Check repository permissions
- **Authentication failure**: Retry authentication process

## Publisher Issues
- **Publisher creation**: Create publisher if needed
- **Publisher selection**: Choose correct publisher
- **Publisher verification**: Verify publisher information
- **Publisher access**: Check publisher permissions

## Node Issues
- **Node information**: Verify node details
- **Repository link**: Check GitHub repository
- **Migration status**: Confirm migration from legacy system
- **Claim eligibility**: Verify claim eligibility
```

#### Debugging Steps
```markdown
# Debugging Claim Issues

## Authentication Debugging
1. **Check GitHub account**: Verify correct account
2. **Repository access**: Confirm admin privileges
3. **Authentication status**: Check login status
4. **Permission verification**: Verify repository permissions

## Publisher Debugging
1. **Publisher creation**: Create publisher if needed
2. **Publisher selection**: Choose correct publisher
3. **Publisher verification**: Verify publisher information
4. **Publisher access**: Check publisher permissions

## Node Debugging
1. **Node information**: Verify node details
2. **Repository link**: Check GitHub repository
3. **Migration status**: Confirm migration status
4. **Claim eligibility**: Verify claim eligibility
```

### Important Notes

#### Claim Benefits
```markdown
# Claim Benefits

## For Developers
- **Node ownership**: Full control over node
- **Version management**: Manage node versions
- **Registry publishing**: Publish to registry
- **Community engagement**: Handle user feedback

## For Users
- **Verified ownership**: Confirmed node ownership
- **Quality assurance**: Original author management
- **Security**: Verified node security
- **Community oversight**: Community feedback and ratings
```

#### Migration Timeline
```markdown
# Migration Timeline

## Legacy to Registry
1. **Legacy system**: Original ComfyUI Manager
2. **Registry migration**: Move to Comfy Registry
3. **Unclaimed status**: Waiting for author claim
4. **Claim process**: Author claims ownership
5. **Registry integration**: Full registry capabilities

## Claim Process
1. **Account setup**: 5-10 minutes
2. **Publisher creation**: 5-10 minutes
3. **Authentication**: 2-3 minutes
4. **Verification**: 1-2 minutes
5. **Claim completion**: 1-2 minutes
```

## ComfyUI Registry - Security and Quality Standards

### Purpose
The ComfyUI Registry maintains strict security and quality standards to ensure safe, functional, and valuable custom nodes for the community. These standards protect users from malicious code, ensure compatibility, and maintain high-quality development practices.

### Features
- **Security standards**: Protection against malicious code and vulnerabilities
- **Quality requirements**: Functional, documented, and maintained nodes
- **Community value**: Valuable functionality for ComfyUI users
- **Legal compliance**: Adherence to applicable laws and regulations
- **Fork guidelines**: Clear differentiation and significant improvements

### Base Standards

#### 1. Community Value
```markdown
# Community Value Requirements

## Requirements
- **Valuable functionality**: Must provide useful features to ComfyUI community
- **Clear purpose**: Well-defined functionality and use cases
- **User benefit**: Genuine value for ComfyUI users
- **Community contribution**: Positive impact on ComfyUI ecosystem

## Prohibited Practices
- **Excessive self-promotion**: Avoid excessive marketing content
- **Impersonation**: No misleading behavior or false claims
- **Malicious behavior**: No harmful or destructive functionality
- **Spam content**: No irrelevant or low-quality content

## Permitted Self-Promotion
- **Settings menu**: Self-promotion allowed in designated settings section
- **Useful functionality**: Top and side menus should contain useful features only
- **Community benefit**: Focus on community value over self-promotion
```

#### 2. Node Compatibility
```markdown
# Node Compatibility Requirements

## Compatibility Standards
- **No interference**: Don't interfere with other custom nodes
- **Installation safety**: Don't break other nodes' installation
- **Update compatibility**: Don't interfere with other nodes' updates
- **Removal safety**: Don't break other nodes' removal

## Dependency Management
- **Clear warnings**: Display warnings when dependent functionality is used
- **Example workflows**: Provide workflows demonstrating required nodes
- **Dependency documentation**: Document all dependencies clearly
- **Fallback handling**: Handle missing dependencies gracefully
```

#### 3. Legal Compliance
```markdown
# Legal Compliance Requirements

## Compliance Standards
- **Applicable laws**: Must comply with all applicable laws and regulations
- **Jurisdiction compliance**: Follow laws in relevant jurisdictions
- **Data protection**: Comply with data protection regulations
- **Intellectual property**: Respect intellectual property rights

## Legal Considerations
- **Copyright compliance**: Respect copyright and licensing
- **Privacy regulations**: Follow privacy and data protection laws
- **Export controls**: Comply with export control regulations
- **Terms of service**: Follow platform terms of service
```

#### 4. Quality Requirements
```markdown
# Quality Requirements

## Quality Standards
- **Fully functional**: Nodes must work as intended
- **Well documented**: Comprehensive documentation and examples
- **Actively maintained**: Regular updates and bug fixes
- **User support**: Provide user support and feedback

## Documentation Requirements
- **README**: Clear installation and usage instructions
- **API documentation**: Document all functions and parameters
- **Example workflows**: Provide working example workflows
- **Troubleshooting**: Include common issues and solutions
```

#### 5. Fork Guidelines
```markdown
# Fork Guidelines

## Fork Requirements
- **Distinct names**: Must have clearly different names from original
- **Significant differences**: Provide substantial functionality or code changes
- **Improvement justification**: Clear reasons for forking
- **Original attribution**: Proper attribution to original work

## Fork Standards
- **Unique functionality**: Add new features or improvements
- **Code differentiation**: Substantial code changes
- **User benefit**: Clear benefits over original
- **Community value**: Provide additional value to community
```

### Security Standards

#### eval/exec Calls Policy
```markdown
# eval/exec Calls Policy

## Prohibition
- **eval() function**: Prohibited in custom nodes
- **exec() function**: Prohibited in custom nodes
- **Dynamic code execution**: No arbitrary code execution
- **User input processing**: No eval/exec with user inputs

## Security Risks
- **Remote Code Execution (RCE)**: Potential for arbitrary code execution
- **Workflow exploitation**: Workflows can be exploited for attacks
- **Malicious code execution**: Risk of harmful code execution
- **Security vulnerabilities**: Multiple attack vectors

## Attack Vectors
- **Keylogging**: Potential for keylogging attacks
- **Ransomware**: Risk of ransomware attacks
- **Data theft**: Potential for data theft
- **System compromise**: Risk of system compromise
```

#### subprocess for pip install Policy
```markdown
# subprocess for pip install Policy

## Prohibition
- **Runtime installation**: No subprocess calls for pip install
- **Package installation**: No dynamic package installation
- **Dependency management**: Use ComfyUI Manager instead
- **Manual installation**: Avoid manual package installation

## Reasoning
- **ComfyUI Manager**: Centralized dependency management
- **Security improvement**: Prevents supply chain attacks
- **User experience**: Eliminates multiple ComfyUI reloads
- **Dependency tracking**: Better dependency management

## Best Practices
- **requirements.txt**: Use requirements.txt for dependencies
- **ComfyUI Manager**: Let users install dependencies through manager
- **Documentation**: Document all dependencies clearly
- **Fallback handling**: Handle missing dependencies gracefully
```

#### Code Obfuscation Policy
```markdown
# Code Obfuscation Policy

## Prohibition
- **Obfuscated code**: No obfuscated or minified code
- **Code hiding**: No attempts to hide code functionality
- **Encrypted code**: No encrypted or encoded code
- **Unreadable code**: No intentionally unreadable code

## Reasoning
- **Code review**: Impossible to review obfuscated code
- **Security risk**: Obfuscated code likely to be malicious
- **Transparency**: Code must be transparent and reviewable
- **Community trust**: Obfuscated code erodes community trust

## Requirements
- **Readable code**: Code must be human-readable
- **Clear functionality**: Code purpose must be clear
- **Reviewable**: Code must be reviewable by community
- **Transparent**: No hidden or obfuscated functionality
```

### Implementation Guidelines

#### Security Best Practices
```python
# Security-compliant node example
import os
import json
from pathlib import Path

class SecureCustomNode:
    """
    Example of a security-compliant custom node
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("STRING", {"multiline": True})
            }
        }
    
    @classmethod
    def execute(cls, input_data):
        # Safe operations only
        try:
            # No eval/exec calls
            # No subprocess calls
            # No obfuscated code
            result = cls.process_data_safely(input_data)
            return (result,)
        except Exception as e:
            # Proper error handling
            return (f"Error: {str(e)}",)
    
    @classmethod
    def process_data_safely(cls, data):
        # Only safe data processing
        # No external dependencies
        # No system calls
        return data.upper()
```

#### Quality Implementation
```python
# Quality-compliant node example
class QualityCustomNode:
    """
    Example of a quality-compliant custom node
    """
    
    # Version information
    VERSION = "1.0.0"
    AUTHOR = "Your Name"
    DESCRIPTION = "A quality-compliant custom node"
    CATEGORY = "Custom/Quality"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {"default": "Hello World"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    
    @classmethod
    def get_quality_info(cls):
        return {
            "version": cls.VERSION,
            "author": cls.AUTHOR,
            "description": cls.DESCRIPTION,
            "documentation": "https://github.com/yourusername/your-node#readme",
            "examples": "https://github.com/yourusername/your-node/tree/main/examples",
            "support": "https://github.com/yourusername/your-node/issues"
        }
    
    def process(self, input):
        # Well-documented functionality
        # Clear error handling
        # Proper return values
        return (input,)
```

### Compliance Checklist

#### Security Compliance
```markdown
# Security Compliance Checklist

## Code Security
- [ ] No eval/exec calls
- [ ] No subprocess calls for pip install
- [ ] No obfuscated code
- [ ] No malicious functionality
- [ ] Safe data processing
- [ ] Proper error handling

## Dependency Security
- [ ] Use requirements.txt
- [ ] Document all dependencies
- [ ] No dynamic package installation
- [ ] Safe dependency management
- [ ] Fallback handling for missing dependencies
```

#### Quality Compliance
```markdown
# Quality Compliance Checklist

## Functionality
- [ ] Fully functional node
- [ ] Well-documented code
- [ ] Clear error handling
- [ ] Proper return values
- [ ] User-friendly interface

## Documentation
- [ ] Comprehensive README
- [ ] API documentation
- [ ] Example workflows
- [ ] Troubleshooting guide
- [ ] User support information
```

#### Community Value Compliance
```markdown
# Community Value Compliance Checklist

## Value Proposition
- [ ] Clear functionality
- [ ] User benefit
- [ ] Community contribution
- [ ] Positive impact
- [ ] Useful features

## Content Quality
- [ ] No excessive self-promotion
- [ ] No misleading behavior
- [ ] No malicious content
- [ ] High-quality content
- [ ] Community-focused
```

### Best Practices

#### Development Standards
```markdown
# Development Best Practices

## Code Quality
- **Readable code**: Write clear, readable code
- **Documentation**: Document all functions and classes
- **Error handling**: Implement proper error handling
- **Testing**: Test functionality thoroughly

## Security Practices
- **Safe operations**: Use only safe operations
- **Input validation**: Validate all user inputs
- **Error handling**: Handle errors gracefully
- **Dependency management**: Manage dependencies safely
```

#### Community Engagement
```markdown
# Community Engagement Best Practices

## User Support
- **Documentation**: Provide comprehensive documentation
- **Examples**: Include working examples
- **Support**: Provide user support
- **Feedback**: Respond to user feedback

## Community Contribution
- **Value**: Provide genuine value to community
- **Quality**: Maintain high quality standards
- **Updates**: Regular updates and improvements
- **Collaboration**: Collaborate with community
```

### Important Notes

#### Registry Benefits
```markdown
# Registry Benefits

## For Developers
- **Quality assurance**: Maintained quality standards
- **Security**: Security-verified nodes
- **Community trust**: Trusted by community
- **Support**: Community support and feedback

## For Users
- **Safe installation**: Security-verified nodes
- **Quality assurance**: High-quality nodes
- **Community oversight**: Community-reviewed nodes
- **Reliable functionality**: Well-tested and maintained nodes
```

#### Compliance Timeline
```markdown
# Compliance Timeline

## Initial Review
1. **Security scan**: Automatic security scanning
2. **Quality review**: Manual quality review
3. **Compliance check**: Standards compliance verification
4. **Approval**: Registry approval process

## Ongoing Compliance
1. **Regular updates**: Maintain compliance with updates
2. **Security monitoring**: Ongoing security monitoring
3. **Quality maintenance**: Maintain quality standards
4. **Community feedback**: Respond to community feedback
```

## ComfyUI Registry - Custom Node CI/CD

### Purpose
The ComfyUI Registry provides a comprehensive CI/CD (Continuous Integration/Continuous Deployment) system for custom nodes to ensure compatibility, functionality, and quality across different operating systems and configurations. This system helps developers test their nodes before publishing and prevents breaking changes from reaching users.

### Features
- **Automated testing**: Test custom nodes across different environments
- **Workflow validation**: Run ComfyUI workflows to verify functionality
- **Cross-platform support**: Test on Linux, Mac, and Windows
- **Model management**: Download and test with different models
- **Custom node testing**: Test with other custom nodes for compatibility
- **Results dashboard**: View test results and output files

### Introduction

#### Why CI/CD is Important
```markdown
# CI/CD Importance for Custom Nodes

## Common Issues
- **Breaking changes**: Changes can break ComfyUI or other custom nodes
- **OS compatibility**: Different behavior across operating systems
- **PyTorch configurations**: Various PyTorch versions and configurations
- **Dependency conflicts**: Conflicts with other custom nodes
- **Model compatibility**: Issues with different model types

## CI/CD Benefits
- **Automated testing**: Test changes before publishing
- **Cross-platform validation**: Ensure compatibility across OS
- **Workflow verification**: Validate actual ComfyUI workflows
- **Quality assurance**: Catch issues before they reach users
- **Regression testing**: Prevent breaking existing functionality
```

#### Comfy-Action Overview
```markdown
# Comfy-Action Overview

## Capabilities
- **Workflow execution**: Run ComfyUI workflow.json files
- **Model downloading**: Automatically download required models
- **Custom node support**: Install and test custom nodes
- **Multi-platform**: Support for Linux, Mac, and Windows
- **GitHub Actions**: Integrate with GitHub Actions workflow

## Features
- **Automated setup**: Automatic ComfyUI setup and configuration
- **Model management**: Download and cache models for testing
- **Custom node installation**: Install custom nodes for testing
- **Workflow execution**: Run complete ComfyUI workflows
- **Output collection**: Collect and upload test results
```

### Getting Started

#### Setup Comfy-Action
```yaml
# .github/workflows/comfy-ci.yml
name: ComfyUI CI/CD Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-comfy-workflow:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Setup ComfyUI with Comfy-Action
        uses: Comfy-Org/comfy-action@main
        with:
          comfyui-version: "latest"
          custom-nodes: |
            ComfyUI-Model_preset_Pilot
            ComfyUI-Manager
          models: |
            https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
      
      - name: Run test workflow
        uses: Comfy-Org/comfy-action@main
        with:
          workflow-file: "test_workflow.json"
          output-dir: "test_outputs"
      
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: test-results-${{ matrix.os }}
          path: test_outputs/
```

#### Workflow Configuration
```json
{
  "workflow_configuration": {
    "comfyui_version": "latest",
    "custom_nodes": [
      "ComfyUI-Model_preset_Pilot",
      "ComfyUI-Manager",
      "ComfyUI-Extra-Models"
    ],
    "models": [
      "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
      "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt"
    ],
    "test_workflows": [
      "test_basic_workflow.json",
      "test_advanced_workflow.json",
      "test_error_handling.json"
    ]
  }
}
```

### Advanced Configuration

#### Custom Node Testing
```yaml
# Advanced CI/CD configuration
name: Advanced ComfyUI Testing

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-custom-node:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Setup ComfyUI
        uses: Comfy-Org/comfy-action@main
        with:
          comfyui-version: "latest"
          custom-nodes: |
            ComfyUI-Model_preset_Pilot
            ComfyUI-Manager
            ComfyUI-Extra-Models
            ComfyUI-AnimateDiff-Evolved
          models: |
            https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
            https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt
      
      - name: Test basic functionality
        uses: Comfy-Org/comfy-action@main
        with:
          workflow-file: "tests/basic_test.json"
          output-dir: "test_outputs/basic"
      
      - name: Test advanced features
        uses: Comfy-Org/comfy-action@main
        with:
          workflow-file: "tests/advanced_test.json"
          output-dir: "test_outputs/advanced"
      
      - name: Test error handling
        uses: Comfy-Org/comfy-action@main
        with:
          workflow-file: "tests/error_test.json"
          output-dir: "test_outputs/error"
      
      - name: Upload all test results
        uses: actions/upload-artifact@v3
        with:
          name: comprehensive-test-results
          path: test_outputs/
```

#### Multi-Platform Testing
```yaml
# Multi-platform testing configuration
name: Multi-Platform ComfyUI Testing

on:
  push:
    branches: [ main ]

jobs:
  test-multi-platform:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Setup ComfyUI
        uses: Comfy-Org/comfy-action@main
        with:
          comfyui-version: "latest"
          custom-nodes: |
            ComfyUI-Model_preset_Pilot
            ComfyUI-Manager
          models: |
            https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
      
      - name: Run platform-specific tests
        uses: Comfy-Org/comfy-action@main
        with:
          workflow-file: "tests/platform_test.json"
          output-dir: "test_outputs/${{ matrix.os }}-${{ matrix.python-version }}"
      
      - name: Upload platform results
        uses: actions/upload-artifact@v3
        with:
          name: test-results-${{ matrix.os }}-${{ matrix.python-version }}
          path: test_outputs/${{ matrix.os }}-${{ matrix.python-version }}/
```

### Test Workflow Examples

#### Basic Test Workflow
```json
{
  "basic_test_workflow": {
    "1": {
      "class_type": "CheckpointLoaderSimple",
      "inputs": {
        "ckpt_name": "v1-5-pruned-emaonly.ckpt"
      }
    },
    "2": {
      "class_type": "CLIPTextEncode",
      "inputs": {
        "text": "a beautiful landscape",
        "clip": ["1", 1]
      }
    },
    "3": {
      "class_type": "CLIPTextEncode",
      "inputs": {
        "text": "blurry, low quality",
        "clip": ["1", 1]
      }
    },
    "4": {
      "class_type": "KSampler",
      "inputs": {
        "seed": 12345,
        "steps": 20,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "model": ["1", 0],
        "positive": ["2", 0],
        "negative": ["3", 0],
        "latent_image": ["5", 0]
      }
    },
    "5": {
      "class_type": "EmptyLatentImage",
      "inputs": {
        "width": 512,
        "height": 512,
        "batch_size": 1
      }
    },
    "6": {
      "class_type": "VAEDecode",
      "inputs": {
        "samples": ["4", 0],
        "vae": ["1", 2]
      }
    },
    "7": {
      "class_type": "SaveImage",
      "inputs": {
        "filename_prefix": "test_output",
        "images": ["6", 0]
      }
    }
  }
}
```

#### Custom Node Test Workflow
```json
{
  "custom_node_test_workflow": {
    "1": {
      "class_type": "CheckpointLoaderSimple",
      "inputs": {
        "ckpt_name": "v1-5-pruned-emaonly.ckpt"
      }
    },
    "2": {
      "class_type": "ModelPresetLoader",
      "inputs": {
        "model": ["1", 0],
        "preset_name": "realistic"
      }
    },
    "3": {
      "class_type": "CLIPTextEncode",
      "inputs": {
        "text": "a beautiful landscape",
        "clip": ["1", 1]
      }
    },
    "4": {
      "class_type": "CLIPTextEncode",
      "inputs": {
        "text": "blurry, low quality",
        "clip": ["1", 1]
      }
    },
    "5": {
      "class_type": "KSampler",
      "inputs": {
        "seed": 12345,
        "steps": 20,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "model": ["2", 0],
        "positive": ["3", 0],
        "negative": ["4", 0],
        "latent_image": ["6", 0]
      }
    },
    "6": {
      "class_type": "EmptyLatentImage",
      "inputs": {
        "width": 512,
        "height": 512,
        "batch_size": 1
      }
    },
    "7": {
      "class_type": "VAEDecode",
      "inputs": {
        "samples": ["5", 0],
        "vae": ["1", 2]
      }
    },
    "8": {
      "class_type": "SaveImage",
      "inputs": {
        "filename_prefix": "custom_node_test",
        "images": ["7", 0]
      }
    }
  }
}
```

### Results and Dashboard

#### CI/CD Dashboard
```markdown
# CI/CD Dashboard Features

## Results Display
- **Test results**: View test execution results
- **Output files**: Download generated output files
- **Error logs**: View detailed error information
- **Performance metrics**: Monitor test performance
- **Platform comparison**: Compare results across platforms

## Dashboard Benefits
- **Visual feedback**: See test results before committing
- **Quality assurance**: Ensure changes don't break functionality
- **Regression testing**: Prevent breaking existing features
- **Cross-platform validation**: Verify compatibility across OS
- **Automated validation**: Reduce manual testing effort
```

#### Output Management
```yaml
# Output management configuration
name: Output Management

on:
  push:
    branches: [ main ]

jobs:
  test-with-outputs:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Setup ComfyUI
        uses: Comfy-Org/comfy-action@main
        with:
          comfyui-version: "latest"
          custom-nodes: |
            ComfyUI-Model_preset_Pilot
          models: |
            https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
      
      - name: Run test with outputs
        uses: Comfy-Org/comfy-action@main
        with:
          workflow-file: "test_workflow.json"
          output-dir: "test_outputs"
          save-outputs: true
      
      - name: Upload test outputs
        uses: actions/upload-artifact@v3
        with:
          name: test-outputs
          path: test_outputs/
          retention-days: 30
      
      - name: Upload to dashboard
        uses: Comfy-Org/upload-to-dashboard@main
        with:
          dashboard-url: "https://ci.comfy.org"
          api-key: ${{ secrets.DASHBOARD_API_KEY }}
          test-results: "test_outputs/"
```

### Best Practices

#### CI/CD Best Practices
```markdown
# CI/CD Best Practices

## Test Coverage
- **Basic functionality**: Test core node functionality
- **Error handling**: Test error conditions and edge cases
- **Integration testing**: Test with other custom nodes
- **Performance testing**: Monitor performance and memory usage
- **Regression testing**: Ensure changes don't break existing functionality

## Workflow Design
- **Modular tests**: Create separate test workflows for different features
- **Realistic scenarios**: Use realistic test scenarios and data
- **Error simulation**: Test error conditions and recovery
- **Performance monitoring**: Monitor execution time and resource usage
- **Output validation**: Verify output quality and correctness
```

#### Testing Strategy
```markdown
# Testing Strategy

## Test Types
- **Unit tests**: Test individual node functionality
- **Integration tests**: Test node interactions with ComfyUI
- **Workflow tests**: Test complete ComfyUI workflows
- **Performance tests**: Monitor performance and resource usage
- **Compatibility tests**: Test across different platforms and configurations

## Test Organization
- **Test structure**: Organize tests by functionality and complexity
- **Test data**: Use realistic test data and scenarios
- **Test isolation**: Ensure tests don't interfere with each other
- **Test cleanup**: Clean up test artifacts and temporary files
- **Test documentation**: Document test purpose and expected results
```

### Important Notes

#### CI/CD Benefits
```markdown
# CI/CD Benefits

## For Developers
- **Automated testing**: Reduce manual testing effort
- **Quality assurance**: Catch issues before publishing
- **Cross-platform validation**: Ensure compatibility across OS
- **Regression prevention**: Prevent breaking existing functionality
- **Continuous integration**: Integrate changes safely

## For Users
- **Reliable nodes**: Higher quality and reliability
- **Compatibility assurance**: Works across different platforms
- **Quality validation**: Tested and validated functionality
- **Error prevention**: Fewer bugs and issues
- **Performance optimization**: Optimized for different environments
```

#### Implementation Timeline
```markdown
# Implementation Timeline

## Setup Phase
1. **Repository setup**: 5-10 minutes
2. **Workflow configuration**: 10-15 minutes
3. **Test workflow creation**: 15-30 minutes
4. **Initial testing**: 5-10 minutes

## Ongoing Maintenance
1. **Test updates**: 5-10 minutes per update
2. **Workflow maintenance**: 10-15 minutes
3. **Test execution**: 5-10 minutes
4. **Results review**: 5-10 minutes
```

## ComfyUI Registry - pyproject.toml Specifications

### Purpose
The `pyproject.toml` file is the configuration file for ComfyUI custom nodes in the registry. It contains metadata about the node, dependencies, compatibility requirements, and registry-specific information. This file is essential for publishing custom nodes to the ComfyUI Registry.

### Features
- **Node metadata**: Name, version, description, and licensing information
- **Dependency management**: Python and ComfyUI version requirements
- **Registry configuration**: Publisher information and display settings
- **Compatibility specification**: OS and GPU accelerator support
- **URL management**: Repository, documentation, and bug tracker links

### [project] Section

#### name (required)
```toml
# Node ID specification
[project]
name = "super-resolution-node"  # ✅ Good: Simple and clear
name = "image-processor"        # ✅ Good: Describes functionality
name = "ComfyUI-enhancer"       # ❌ Bad: Includes ComfyUI
name = "123-tool"               # ❌ Bad: Starts with number
```

**Requirements:**
- Must be less than 100 characters
- Can only contain alphanumeric characters, hyphens, underscores, and periods
- Cannot have consecutive special characters
- Cannot start with a number or special character
- Case-insensitive comparison

**Best Practices:**
- Use a short, descriptive name
- Don't include "ComfyUI" in the name
- Make it memorable and easy to type

#### version (required)
```toml
# Semantic versioning
[project]
version = "1.0.0"    # Initial release
version = "1.1.0"    # Added new features
version = "1.1.1"    # Bug fix
version = "2.0.0"    # Breaking changes
```

**Version Format:**
- **X (MAJOR)**: Breaking changes
- **Y (MINOR)**: New features (backwards compatible)
- **Z (PATCH)**: Bug fixes

#### license (optional)
```toml
# License specification
[project]
# File reference
license = { file = "LICENSE" }     # ✅ Points to LICENSE file
license = { file = "LICENSE.txt" } # ✅ Points to LICENSE.txt

# License name
license = { text = "MIT License" }  # ✅ Correct format
license = { text = "Apache-2.0" }   # ✅ Correct format
```

**Common licenses:** MIT, GPL, Apache

#### description (recommended)
```toml
# Node description
[project]
description = "A super resolution node for enhancing image quality"
```

#### repository (required)
```toml
# Repository URL
[project.urls]
Repository = "https://github.com/username/repository"
```

#### urls (recommended)
```toml
# Additional URLs
[project.urls]
Documentation = "https://github.com/username/repository/wiki"
"Bug Tracker" = "https://github.com/username/repository/issues"
```

#### requires-python (recommended)
```toml
# Python version requirements
[project]
requires-python = ">=3.8"        # Python 3.8 or higher
requires-python = ">=3.8,<3.11"  # Python 3.8 up to (but not including) 3.11
```

#### Frontend Version Compatibility (optional)
```toml
# ComfyUI frontend version compatibility
[project]
dependencies = [
    "comfyui-frontend-package>=1.20.0"       # Requires frontend 1.20.0 or newer
    "comfyui-frontend-package<=1.21.6"       # Restricts to frontend versions up to 1.21.6
    "comfyui-frontend-package>=1.19,<1.22"   # Works with frontend 1.19 to 1.21.x
    "comfyui-frontend-package~=1.20.0"       # Compatible with 1.20.x but not 1.21.0
    "comfyui-frontend-package!=1.21.3"       # Works with any version except 1.21.3
]
```

**Use cases:**
- Your custom node uses frontend APIs introduced in specific versions
- You've identified incompatibilities with certain frontend versions
- Your node requires specific UI features only available in newer versions

#### classifiers (recommended)
```toml
# Operating system and GPU compatibility
[project]
classifiers = [
    # OS compatibility
    "Operating System :: OS Independent",  # Works on all operating systems
    "Operating System :: Microsoft :: Windows",  # Windows specific
    "Operating System :: POSIX :: Linux",  # Linux specific
    "Operating System :: MacOS",  # macOS specific
    
    # GPU Accelerator support
    "Environment :: GPU :: NVIDIA CUDA",    # NVIDIA CUDA support
    "Environment :: GPU :: AMD ROCm",       # AMD ROCm support
    "Environment :: GPU :: Intel Arc",      # Intel Arc support
    "Environment :: NPU :: Huawei Ascend",  # Huawei Ascend support
    "Environment :: GPU :: Apple Metal",    # Apple Metal support
]
```

### [tool.comfy] Section

#### PublisherId (required)
```toml
# Publisher identifier
[tool.comfy]
PublisherId = "john-doe"        # ✅ Matches GitHub username
PublisherId = "image-wizard"    # ✅ Unique identifier
```

#### DisplayName (optional)
```toml
# User-friendly display name
[tool.comfy]
DisplayName = "Super Resolution Node"
```

#### Icon (optional)
```toml
# Node icon URL
[tool.comfy]
Icon = "https://raw.githubusercontent.com/username/repo/main/icon.png"
```

**Requirements:**
- File types: SVG, PNG, JPG, or GIF
- Maximum resolution: 400px × 400px
- Aspect ratio should be square

#### Banner (optional)
```toml
# Banner image URL
[tool.comfy]
Banner = "https://raw.githubusercontent.com/username/repo/main/banner.png"
```

**Requirements:**
- File types: SVG, PNG, JPG, or GIF
- Aspect ratio: 21:9

#### requires-comfyui (optional)
```toml
# ComfyUI version compatibility
[tool.comfy]
requires-comfyui = ">=1.0.0"        # ComfyUI 1.0.0 or higher
requires-comfyui = ">=1.0.0,<2.0.0"  # ComfyUI 1.0.0 up to (but not including) 2.0.0
requires-comfyui = "~=1.0.0"         # Compatible release: version 1.0.0 or newer, but not version 2.0.0
requires-comfyui = "!=1.2.3"         # Any version except 1.2.3
requires-comfyui = ">0.1.3,<1.0.0"   # Greater than 0.1.3 and less than 1.0.0
```

**Supported operators:** `<`, `>`, `<=`, `>=`, `~=`, `<>`, `!=` and ranges

#### includes (optional)
```toml
# Force include specific folders
[tool.comfy]
includes = ['dist']
```

**Use cases:**
- Custom nodes in frontend projects
- Final packaged output folder in .gitignore
- Force include for registry use

### Complete Example

#### Basic Example
```toml
# Basic pyproject.toml example
[project]
name = "super-resolution-node"
version = "1.0.0"
description = "Enhance image quality using advanced super resolution techniques"
license = { file = "LICENSE" }
requires-python = ">=3.8"
dependencies = [
    "comfyui-frontend-package<=1.21.6"  # Frontend version compatibility
]
classifiers = [
    "Operating System :: OS Independent"  # Works on all operating systems
]
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}

[project.urls]
Repository = "https://github.com/username/super-resolution-node"
Documentation = "https://github.com/username/super-resolution-node/wiki"
"Bug Tracker" = "https://github.com/username/super-resolution-node/issues"

[tool.comfy]
PublisherId = "image-wizard"
DisplayName = "Super Resolution Node"
Icon = "https://raw.githubusercontent.com/username/super-resolution-node/main/icon.png"
Banner = "https://raw.githubusercontent.com/username/super-resolution-node/main/banner.png"
requires-comfyui = ">=1.0.0"  # ComfyUI version compatibility
```

#### Advanced Example
```toml
# Advanced pyproject.toml example
[project]
name = "advanced-image-processor"
version = "2.1.3"
description = "Advanced image processing with AI-powered enhancement and real-time preview"
license = { file = "LICENSE" }
requires-python = ">=3.8,<3.12"
dependencies = [
    "torch>=1.9.0",
    "pillow>=8.3.0",
    "numpy>=1.21.0",
    "opencv-python>=4.5.0",
    "comfyui-frontend-package>=1.20.0,<1.22.0"
]
classifiers = [
    "Operating System :: OS Independent",
    "Environment :: GPU :: NVIDIA CUDA",
    "Environment :: GPU :: AMD ROCm",
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    "Topic :: Scientific/Engineering :: Artificial Intelligence"
]
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}

[project.urls]
Repository = "https://github.com/username/advanced-image-processor"
Documentation = "https://github.com/username/advanced-image-processor/wiki"
"Bug Tracker" = "https://github.com/username/advanced-image-processor/issues"
"Homepage" = "https://github.com/username/advanced-image-processor"
"Changelog" = "https://github.com/username/advanced-image-processor/blob/main/CHANGELOG.md"

[tool.comfy]
PublisherId = "advanced-wizard"
DisplayName = "Advanced Image Processor"
Icon = "https://raw.githubusercontent.com/username/advanced-image-processor/main/assets/icon.png"
Banner = "https://raw.githubusercontent.com/username/advanced-image-processor/main/assets/banner.png"
requires-comfyui = ">=1.0.0,<2.0.0"
includes = ['dist', 'assets']
```

### Best Practices

#### Naming Conventions
```markdown
# Naming Best Practices

## Node Names
- **Short and descriptive**: Use clear, concise names
- **No ComfyUI prefix**: Don't include "ComfyUI" in the name
- **Memorable**: Easy to remember and type
- **Unique**: Ensure uniqueness in the registry

## Examples
- ✅ "image-processor" - Clear and descriptive
- ✅ "super-resolution" - Describes functionality
- ✅ "ai-upscaler" - Concise and informative
- ❌ "ComfyUI-enhancer" - Includes ComfyUI
- ❌ "123-tool" - Starts with number
```

#### Version Management
```markdown
# Version Management Best Practices

## Semantic Versioning
- **MAJOR (X)**: Breaking changes that require workflow updates
- **MINOR (Y)**: New features that are backward compatible
- **PATCH (Z)**: Bug fixes that are backward compatible

## Version Examples
- `1.0.0`: Initial release
- `1.0.1`: Bug fix (backward compatible)
- `1.1.0`: New feature (backward compatible)
- `2.0.0`: Breaking change (requires workflow updates)
```

#### Dependency Management
```markdown
# Dependency Management Best Practices

## Python Dependencies
- **Version pinning**: Pin specific versions for stability
- **Compatibility**: Ensure compatibility across Python versions
- **Security**: Use trusted packages and update regularly

## ComfyUI Dependencies
- **Frontend compatibility**: Specify frontend version requirements
- **Core compatibility**: Specify ComfyUI core version requirements
- **Testing**: Test with different ComfyUI versions
```

#### Metadata Quality
```markdown
# Metadata Quality Best Practices

## Description
- **Clear purpose**: Describe what the node does
- **Use cases**: Explain when to use the node
- **Features**: Highlight key features and capabilities

## URLs
- **Repository**: Link to source code repository
- **Documentation**: Link to comprehensive documentation
- **Bug tracker**: Link to issue tracker for bug reports
- **Homepage**: Link to project homepage if available
```

### Troubleshooting

#### Common Issues
```markdown
# Common pyproject.toml Issues

## Name Issues
- **Invalid characters**: Use only alphanumeric, hyphens, underscores, periods
- **Length limit**: Keep under 100 characters
- **Special characters**: No consecutive special characters
- **Starting character**: Cannot start with number or special character

## Version Issues
- **Semantic versioning**: Use proper X.Y.Z format
- **Version comparison**: Ensure version ordering is correct
- **Breaking changes**: Increment major version for breaking changes

## Dependency Issues
- **Version conflicts**: Resolve conflicting dependency versions
- **Missing dependencies**: Include all required dependencies
- **Compatibility**: Ensure compatibility with ComfyUI versions
```

#### Validation
```bash
# Validate pyproject.toml
# Check syntax
python -m tomllib pyproject.toml

# Validate with setuptools
python -m setuptools.config.validate pyproject.toml

# Check dependencies
pip check
```

### Important Notes

#### Registry Benefits
```markdown
# Registry Benefits

## For Developers
- **Standardized metadata**: Consistent node information
- **Dependency management**: Clear dependency specifications
- **Version tracking**: Semantic versioning support
- **Compatibility**: OS and GPU compatibility information

## For Users
- **Clear information**: Comprehensive node metadata
- **Dependency awareness**: Know what dependencies are required
- **Compatibility**: Understand system requirements
- **Quality assurance**: Standardized node information
```

#### Implementation Timeline
```markdown
# Implementation Timeline

## Initial Setup
1. **Basic metadata**: 5-10 minutes
2. **Dependency specification**: 10-15 minutes
3. **Registry configuration**: 5-10 minutes
4. **Testing**: 5-10 minutes

## Ongoing Maintenance
1. **Version updates**: 2-3 minutes per update
2. **Dependency updates**: 5-10 minutes
3. **Metadata updates**: 3-5 minutes
4. **Validation**: 2-3 minutes
```

## ComfyUI Node Definition JSON Schema (v2.0)

### Purpose
The ComfyUI Node Definition JSON schema (v2.0) provides a standardized way to define custom nodes using JSON Schema. This schema defines the structure, inputs, outputs, and metadata for ComfyUI custom nodes, ensuring consistency and compatibility across the ecosystem.

### Features
- **Standardized structure**: Consistent node definition format
- **Input specifications**: Detailed input type definitions with validation
- **Output definitions**: Clear output type and structure specifications
- **Metadata support**: Node information, categories, and descriptions
- **Advanced features**: Lazy evaluation, remote data, and dynamic options
- **Type safety**: JSON Schema validation for node definitions

### Schema Overview

#### Core Structure
```json
{
  "inputs": {
    "input_name": {
      "type": "INPUT_TYPE",
      "name": "Display Name",
      "default": "default_value",
      "tooltip": "Help text",
      "hidden": false,
      "advanced": false,
      "lazy": false
    }
  },
  "outputs": [
    {
      "index": 0,
      "name": "Output Name",
      "type": "OUTPUT_TYPE",
      "is_list": false,
      "tooltip": "Output description"
    }
  ],
  "name": "NodeClassName",
  "display_name": "User-Friendly Name",
  "description": "Node description",
  "category": "Node Category",
  "output_node": false,
  "python_module": "module_name",
  "deprecated": false,
  "experimental": false
}
```

### Input Types

#### INT (Integer)
```json
{
  "type": "INT",
  "name": "Steps",
  "default": 20,
  "min": 1,
  "max": 100,
  "step": 1,
  "display": "slider",
  "tooltip": "Number of sampling steps",
  "control_after_generate": true
}
```

**Properties:**
- `default`: Default value (number or array)
- `min`: Minimum value
- `max`: Maximum value
- `step`: Step increment
- `display`: Display type ("slider", "number", "knob")
- `control_after_generate`: Allow control after generation

#### FLOAT (Float)
```json
{
  "type": "FLOAT",
  "name": "CFG Scale",
  "default": 7.0,
  "min": 0.1,
  "max": 20.0,
  "step": 0.1,
  "round": 2,
  "display": "slider",
  "tooltip": "Classifier-free guidance scale"
}
```

**Properties:**
- `default`: Default value (number or array)
- `min`: Minimum value
- `max`: Maximum value
- `step`: Step increment
- `round`: Rounding precision (number or false)
- `display`: Display type ("slider", "number", "knob")

#### BOOLEAN (Boolean)
```json
{
  "type": "BOOLEAN",
  "name": "Enable Feature",
  "default": true,
  "label_on": "On",
  "label_off": "Off",
  "tooltip": "Enable or disable feature"
}
```

**Properties:**
- `default`: Default boolean value
- `label_on`: Label for true state
- `label_off`: Label for false state

#### STRING (String)
```json
{
  "type": "STRING",
  "name": "Prompt",
  "default": "A beautiful landscape",
  "multiline": true,
  "dynamicPrompts": true,
  "placeholder": "Enter your prompt here",
  "tooltip": "Text prompt for generation"
}
```

**Properties:**
- `default`: Default string value
- `multiline`: Allow multiline input
- `dynamicPrompts`: Enable dynamic prompts
- `defaultVal`: Alternative default value
- `placeholder`: Placeholder text

#### COMBO (Combo/Select)
```json
{
  "type": "COMBO",
  "name": "Sampler",
  "default": "euler",
  "options": ["euler", "dpm_2", "dpm_2_ancestral", "lms"],
  "tooltip": "Sampling method"
}
```

**Properties:**
- `default`: Default selected value
- `options`: Array of available options
- `control_after_generate`: Allow control after generation
- `image_upload`: Enable image upload
- `image_folder`: Image folder type ("input", "output", "temp")
- `allow_batch`: Allow batch processing
- `video_upload`: Enable video upload
- `remote`: Remote data configuration

#### Custom Types
```json
{
  "type": "CUSTOM_TYPE",
  "name": "Custom Input",
  "default": null,
  "tooltip": "Custom input type"
}
```

**Properties:**
- `type`: Custom type identifier
- `name`: Display name
- `default`: Default value
- `tooltip`: Help text

### Advanced Input Features

#### Lazy Evaluation
```json
{
  "type": "STRING",
  "name": "Dynamic Input",
  "lazy": true,
  "tooltip": "This input is evaluated lazily"
}
```

#### Hidden Inputs
```json
{
  "type": "STRING",
  "name": "Internal Data",
  "hidden": true,
  "default": "internal_value"
}
```

#### Advanced Inputs
```json
{
  "type": "FLOAT",
  "name": "Advanced Setting",
  "advanced": true,
  "tooltip": "Advanced configuration option"
}
```

#### Remote Data
```json
{
  "type": "COMBO",
  "name": "Remote Options",
  "remote": {
    "route": "/api/options",
    "refresh": 30,
    "response_key": "data",
    "query_params": {
      "category": "samplers"
    },
    "refresh_button": true,
    "control_after_refresh": "first",
    "timeout": 5000,
    "max_retries": 3
  }
}
```

**Remote Properties:**
- `route`: API endpoint URL
- `refresh`: Refresh interval in seconds
- `response_key`: Key to extract data from response
- `query_params`: Query parameters
- `refresh_button`: Show refresh button
- `control_after_refresh`: Control behavior after refresh
- `timeout`: Request timeout
- `max_retries`: Maximum retry attempts

### Output Definitions

#### Basic Output
```json
{
  "index": 0,
  "name": "IMAGE",
  "type": "IMAGE",
  "is_list": false,
  "tooltip": "Generated image"
}
```

#### List Output
```json
{
  "index": 1,
  "name": "IMAGES",
  "type": "IMAGE",
  "is_list": true,
  "tooltip": "List of generated images"
}
```

#### Output with Options
```json
{
  "index": 2,
  "name": "MASK",
  "type": "MASK",
  "is_list": false,
  "options": ["invert", "normalize"],
  "tooltip": "Generated mask with options"
}
```

### Node Metadata

#### Basic Metadata
```json
{
  "name": "MyCustomNode",
  "display_name": "My Custom Node",
  "description": "A custom node for image processing",
  "category": "image/processing",
  "output_node": false,
  "python_module": "my_custom_node"
}
```

#### Advanced Metadata
```json
{
  "name": "AdvancedNode",
  "display_name": "Advanced Processing Node",
  "description": "Advanced image processing with AI features",
  "category": "image/ai",
  "output_node": true,
  "python_module": "advanced_node",
  "deprecated": false,
  "experimental": true
}
```

### Complete Example

#### Basic Node Definition
```json
{
  "inputs": {
    "image": {
      "type": "IMAGE",
      "name": "Image",
      "tooltip": "Input image to process"
    },
    "strength": {
      "type": "FLOAT",
      "name": "Strength",
      "default": 0.5,
      "min": 0.0,
      "max": 1.0,
      "step": 0.01,
      "display": "slider",
      "tooltip": "Processing strength"
    },
    "enable_feature": {
      "type": "BOOLEAN",
      "name": "Enable Feature",
      "default": true,
      "tooltip": "Enable advanced feature"
    },
    "sampler": {
      "type": "COMBO",
      "name": "Sampler",
      "default": "euler",
      "options": ["euler", "dpm_2", "lms"],
      "tooltip": "Sampling method"
    }
  },
  "outputs": [
    {
      "index": 0,
      "name": "IMAGE",
      "type": "IMAGE",
      "is_list": false,
      "tooltip": "Processed image"
    },
    {
      "index": 1,
      "name": "INFO",
      "type": "STRING",
      "is_list": false,
      "tooltip": "Processing information"
    }
  ],
  "name": "ImageProcessor",
  "display_name": "Image Processor",
  "description": "Process images with various effects",
  "category": "image/processing",
  "output_node": false,
  "python_module": "image_processor"
}
```

#### Advanced Node Definition
```json
{
  "inputs": {
    "model": {
      "type": "MODEL",
      "name": "Model",
      "tooltip": "AI model for processing"
    },
    "prompt": {
      "type": "STRING",
      "name": "Prompt",
      "default": "A beautiful landscape",
      "multiline": true,
      "dynamicPrompts": true,
      "placeholder": "Enter your prompt here",
      "tooltip": "Text prompt for generation"
    },
    "steps": {
      "type": "INT",
      "name": "Steps",
      "default": 20,
      "min": 1,
      "max": 100,
      "step": 1,
      "display": "slider",
      "control_after_generate": true,
      "tooltip": "Number of sampling steps"
    },
    "cfg_scale": {
      "type": "FLOAT",
      "name": "CFG Scale",
      "default": 7.0,
      "min": 0.1,
      "max": 20.0,
      "step": 0.1,
      "round": 2,
      "display": "slider",
      "tooltip": "Classifier-free guidance scale"
    },
    "sampler": {
      "type": "COMBO",
      "name": "Sampler",
      "default": "euler",
      "options": ["euler", "dpm_2", "dpm_2_ancestral", "lms"],
      "tooltip": "Sampling method"
    },
    "seed": {
      "type": "INT",
      "name": "Seed",
      "default": 0,
      "min": 0,
      "max": 2147483647,
      "step": 1,
      "display": "number",
      "tooltip": "Random seed for generation"
    },
    "batch_size": {
      "type": "INT",
      "name": "Batch Size",
      "default": 1,
      "min": 1,
      "max": 8,
      "step": 1,
      "display": "slider",
      "tooltip": "Number of images to generate"
    },
    "enable_advanced": {
      "type": "BOOLEAN",
      "name": "Advanced Options",
      "default": false,
      "tooltip": "Enable advanced options"
    },
    "advanced_strength": {
      "type": "FLOAT",
      "name": "Advanced Strength",
      "default": 0.8,
      "min": 0.0,
      "max": 1.0,
      "step": 0.01,
      "display": "slider",
      "advanced": true,
      "tooltip": "Advanced processing strength"
    },
    "remote_options": {
      "type": "COMBO",
      "name": "Remote Options",
      "default": "default",
      "remote": {
        "route": "/api/options",
        "refresh": 30,
        "response_key": "data",
        "query_params": {
          "category": "advanced"
        },
        "refresh_button": true,
        "control_after_refresh": "first",
        "timeout": 5000,
        "max_retries": 3
      },
      "tooltip": "Options loaded from remote API"
    }
  },
  "outputs": [
    {
      "index": 0,
      "name": "IMAGE",
      "type": "IMAGE",
      "is_list": false,
      "tooltip": "Generated image"
    },
    {
      "index": 1,
      "name": "IMAGES",
      "type": "IMAGE",
      "is_list": true,
      "tooltip": "List of generated images"
    },
    {
      "index": 2,
      "name": "INFO",
      "type": "STRING",
      "is_list": false,
      "tooltip": "Generation information"
    }
  ],
  "name": "AdvancedGenerator",
  "display_name": "Advanced Image Generator",
  "description": "Advanced AI image generation with multiple options",
  "category": "image/generation",
  "output_node": true,
  "python_module": "advanced_generator",
  "deprecated": false,
  "experimental": false
}
```

### Best Practices

#### Input Design
```markdown
# Input Design Best Practices

## Input Organization
- **Logical grouping**: Group related inputs together
- **Default values**: Provide sensible defaults
- **Tooltips**: Include helpful descriptions
- **Validation**: Use min/max for numeric inputs
- **Advanced options**: Mark advanced features as advanced

## Input Types
- **Appropriate types**: Use the right input type for the data
- **User experience**: Consider how users will interact with inputs
- **Validation**: Provide clear validation rules
- **Feedback**: Give users feedback on input values
```

#### Output Design
```markdown
# Output Design Best Practices

## Output Clarity
- **Clear names**: Use descriptive output names
- **Type consistency**: Ensure output types match expectations
- **List handling**: Use is_list appropriately
- **Tooltips**: Provide output descriptions

## Output Organization
- **Logical order**: Order outputs logically
- **Index management**: Use consistent indexing
- **Type safety**: Ensure output types are correct
- **Documentation**: Document output structure
```

#### Metadata Quality
```markdown
# Metadata Quality Best Practices

## Node Information
- **Clear names**: Use descriptive node names
- **Helpful descriptions**: Explain what the node does
- **Appropriate categories**: Use correct category classification
- **Version information**: Include version and status information

## Documentation
- **Comprehensive descriptions**: Document all features
- **Usage examples**: Provide usage examples
- **Troubleshooting**: Include troubleshooting information
- **Updates**: Keep documentation current
```

### Validation

#### Schema Validation
```bash
# Validate node definition JSON
# Using JSON Schema validator
python -m jsonschema node_definition.json schema.json

# Using online validator
# Upload to JSON Schema validator website
```

#### Common Validation Issues
```markdown
# Common Validation Issues

## Input Issues
- **Missing required fields**: Ensure all required fields are present
- **Invalid types**: Use correct input types
- **Invalid values**: Check min/max values and constraints
- **Missing tooltips**: Include helpful tooltips

## Output Issues
- **Missing outputs**: Ensure outputs are defined
- **Invalid output types**: Use correct output types
- **Index conflicts**: Avoid duplicate output indices
- **Missing descriptions**: Include output descriptions

## Metadata Issues
- **Missing required fields**: Include all required metadata
- **Invalid categories**: Use valid category names
- **Inconsistent naming**: Ensure consistent naming
- **Missing descriptions**: Include node descriptions
```

### Important Notes

#### Schema Benefits
```markdown
# Schema Benefits

## For Developers
- **Standardized structure**: Consistent node definition format
- **Type safety**: JSON Schema validation
- **Documentation**: Self-documenting node definitions
- **Compatibility**: Ensures compatibility across ComfyUI versions

## For Users
- **Clear interface**: Well-defined input/output structure
- **Validation**: Input validation and error handling
- **Documentation**: Built-in help and tooltips
- **Consistency**: Consistent user experience across nodes
```

#### Implementation Timeline
```markdown
# Implementation Timeline

## Node Definition
1. **Basic structure**: 10-15 minutes
2. **Input definition**: 15-30 minutes
3. **Output definition**: 10-15 minutes
4. **Metadata**: 5-10 minutes

## Advanced Features
1. **Lazy evaluation**: 5-10 minutes
2. **Remote data**: 15-30 minutes
3. **Advanced inputs**: 10-20 minutes
4. **Validation**: 10-15 minutes
```

## Basic Node Structure

### Python Server Side

```python
class MyCustomNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": ("TYPE", {"default": "default_value"}),
            },
            "optional": {
                "optional_input": ("TYPE", {"default": "default_value"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("OUTPUT_TYPE",)
    RETURN_NAMES = ("output_name",)
    FUNCTION = "process"
    CATEGORY = "Custom/My Category"
    
    def process(self, input_name, optional_input=None, unique_id=None, extra_pnginfo=None):
        # Process the inputs
        result = do_something(input_name)
        return (result,)
```

### Key Components

#### INPUT_TYPES
- **required**: Mandatory inputs
- **optional**: Optional inputs  
- **hidden**: System inputs (unique_id, extra_pnginfo)
- **widgets**: Custom UI widgets (buttons, sliders, etc.)

#### Widget Types
- **STRING**: Text input
- **INT**: Integer input
- **FLOAT**: Float input
- **BOOLEAN**: Checkbox (acts as button when clicked)
- **IMAGE**: Image tensor
- **MODEL**: Model object
- **CONDITIONING**: Conditioning data
- **LATENT**: Latent space data

#### RETURN_TYPES
- Define output types in order
- Must match the number of return values

#### RETURN_NAMES
- Human-readable names for outputs
- Must match RETURN_TYPES count

#### FUNCTION
- Name of the method that processes inputs
- Must be a method of the class

#### CATEGORY
- Where the node appears in the UI
- Format: "Category/Subcategory"

## Widget Implementation

### Boolean Widget (Button-like)
```python
"widgets": {
    "my_button": ("BOOLEAN", {"default": False, "label": "Click Me"}),
}
```

### String Widget
```python
"widgets": {
    "my_text": ("STRING", {"default": "", "multiline": True}),
}
```

### Choice Widget
```python
"widgets": {
    "my_choice": ("STRING", {"default": "option1", "choices": ["option1", "option2"]}),
}
```

## JavaScript Client Side

### Basic Extension
```javascript
app.registerExtension({
    name: "MyExtension",
    async nodeCreated(node) {
        if (node.comfyClass === "MyCustomNode") {
            // Custom UI setup
        }
    }
});
```

### Custom Widgets
```javascript
function setupCustomUI(node) {
    // Create custom elements
    const button = document.createElement("button");
    button.textContent = "Custom Button";
    
    // Add to node
    const container = node.widgets[0].options.el.parentElement;
    container.appendChild(button);
    
    // Handle events
    button.addEventListener("click", () => {
        // Handle button click
    });
}
```

## ComfyUI Datatypes - Complete Reference

### Purpose
Datatypes provide **strong typing** on the client side, preventing workflows from passing wrong data types between nodes. The JavaScript client generally prevents connections between incompatible types.

### Primitive Datatypes

#### INT
```python
"my_int": ("INT", {"default": 10, "min": 0, "max": 100})
```
- **Python type**: `int`
- **Required params**: `default`
- **Optional params**: `min`, `max`

#### FLOAT
```python
"my_float": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
```
- **Python type**: `float`
- **Required params**: `default`
- **Optional params**: `min`, `max`, `step`

#### STRING
```python
"my_string": ("STRING", {"default": "hello", "multiline": True})
```
- **Python type**: `str`
- **Required params**: `default`
- **Optional params**: `multiline`, `placeholder`

#### BOOLEAN
```python
"my_bool": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"})
```
- **Python type**: `bool`
- **Required params**: `default`
- **Optional params**: `label_on`, `label_off`

### Tensor Datatypes

#### IMAGE
```python
"image": ("IMAGE", {})
```
- **Python type**: `torch.Tensor`
- **Shape**: `[B, H, W, C]` (batch, height, width, channels)
- **Channels**: Generally 3 for RGB
- **Range**: 0-1 (float32)

#### LATENT
```python
"latent": ("LATENT", {})
```
- **Python type**: `dict`
- **Key**: `samples` (torch.Tensor)
- **Shape**: `[B, C, H, W]` (batch, channels, height, width)
- **Channels**: Generally 4 for SD models
- **Size**: Height/width are 1/8 of image size

#### MASK
```python
"mask": ("MASK", {})
```
- **Python type**: `torch.Tensor`
- **Shape**: `[H, W]` or `[B, C, H, W]`

#### AUDIO
```python
"audio": ("AUDIO", {})
```
- **Python type**: `dict`
- **Keys**: `waveform` (torch.Tensor), `sample_rate` (int)
- **Shape**: `[B, C, T]` (batch, channels, time)
- **Channels**: 1 for mono, 2 for stereo

### Special Datatypes

#### COMBO (Dropdown)
```python
"my_choice": (["option1", "option2", "option3"], {})
```
- **Python type**: `str`
- **Format**: List of options, first is default
- **Dynamic**: Can be generated at runtime
- **Example**: `"ckpt_name": (folder_paths.get_filename_list("checkpoints"), )`

#### NOISE
```python
"noise": ("NOISE", {})
```
- **Python type**: Custom object
- **Method**: `generate_noise(self, input_latent: Tensor) -> Tensor`
- **Property**: `seed: Optional[int]`
- **Purpose**: Noise source for sampling

#### SAMPLER
```python
"sampler": ("SAMPLER", {})
```
- **Python type**: Custom object
- **Method**: `sample` method
- **Purpose**: Sampling algorithm

#### SIGMAS
```python
"sigmas": ("SIGMAS", {})
```
- **Python type**: `torch.Tensor`
- **Shape**: `[steps+1]`
- **Purpose**: Noise levels for each sampling step
- **Example**: `[14.6146, 10.7468, 8.0815, ..., 0.0000]`

#### GUIDER
```python
"guider": ("GUIDER", {})
```
- **Python type**: Callable object
- **Method**: `__call__(*args, **kwargs)`
- **Input**: Noisy latents `[B, C, H, W]`
- **Output**: Noise prediction `[B, C, H, W]`

### Model Datatypes

#### MODEL
```python
"model": ("MODEL", {})
```
- **Purpose**: Stable diffusion model
- **Python type**: Custom model object

#### CLIP
```python
"clip": ("CLIP", {})
```
- **Purpose**: CLIP text encoder
- **Python type**: Custom CLIP object

#### VAE
```python
"vae": ("VAE", {})
```
- **Purpose**: Variational Autoencoder
- **Python type**: Custom VAE object

#### CONDITIONING
```python
"conditioning": ("CONDITIONING", {})
```
- **Purpose**: Text conditioning data
- **Python type**: List of conditioning objects

### Additional Parameters

#### Standard Parameters
| Key | Description | Applicable Types |
|-----|-------------|------------------|
| `default` | Default value | All types |
| `min` | Minimum value | INT, FLOAT |
| `max` | Maximum value | INT, FLOAT |
| `step` | Increment amount | FLOAT |
| `label_on` | Label when True | BOOLEAN |
| `label_off` | Label when False | BOOLEAN |
| `multiline` | Multiline text box | STRING |
| `placeholder` | Placeholder text | STRING |

#### Advanced Parameters
| Key | Description | Purpose |
|-----|-------------|---------|
| `defaultInput` | Default to input socket | Widget override |
| `forceInput` | Force input socket | Widget override |
| `dynamicPrompts` | Evaluate dynamic prompts | STRING |
| `lazy` | Lazy evaluation | All types |
| `rawLink` | Receive link instead of value | Node expansion |

### Examples

#### Basic Input Types
```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "image": ("IMAGE", {}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
            "enabled": ("BOOLEAN", {"default": True}),
            "mode": (["fast", "quality", "balanced"], {}),
        },
        "optional": {
            "text": ("STRING", {"default": "", "multiline": True}),
            "mask": ("MASK", {}),
        }
    }
```

#### Advanced Input Types
```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "model": ("MODEL", {}),
            "clip": ("CLIP", {}),
            "conditioning": ("CONDITIONING", {}),
            "latent": ("LATENT", {}),
            "noise": ("NOISE", {}),
            "sampler": ("SAMPLER", {}),
            "sigmas": ("SIGMAS", {}),
        }
    }
```

### Custom Datatypes

#### Defining Custom Types
```python
# Custom datatype example
class CustomDataType:
    def __init__(self, value):
        self.value = value
    
    def process(self):
        return self.value * 2

# Usage in INPUT_TYPES
"custom_input": ("CUSTOM", {})
```

#### Type Validation
```python
@classmethod
def VALIDATE_INPUTS(cls, custom_input):
    if not isinstance(custom_input, CustomDataType):
        return "Input must be CustomDataType"
    return True
```

### Best Practices

#### Type Safety
- **Use appropriate types**: Choose the most specific type
- **Validate inputs**: Use VALIDATE_INPUTS for custom types
- **Handle edge cases**: Consider None values and type mismatches

#### Performance
- **Minimal tensors**: Avoid unnecessary tensor operations
- **Efficient processing**: Use appropriate tensor shapes
- **Memory management**: Handle large tensors carefully

#### User Experience
- **Clear labels**: Use descriptive parameter names
- **Sensible defaults**: Choose reasonable default values
- **Input validation**: Provide helpful error messages

## Data Types

### Common Input Types
- **MODEL**: Model object
- **IMAGE**: Image tensor [batch, height, width, channels]
- **LATENT**: Latent space tensor
- **CONDITIONING**: Text conditioning
- **CLIP**: CLIP model
- **VAE**: VAE model
- **STRING**: Text string
- **INT**: Integer
- **FLOAT**: Float
- **BOOLEAN**: Boolean

### Tensor Shapes
- **IMAGE**: `[batch, height, width, channels]` (float32, 0-1 range)
- **LATENT**: `[batch, channels, height, width]` (float32)
- **CONDITIONING**: List of conditioning objects

## Working with Images, Latents, and Masks

### Key Concepts
- **torch.Tensor**: Core data structure for all tensor types
- **Single output**: Return `(tensor,)` not `(tensor)` - trailing comma required
- **Shape awareness**: Understanding tensor dimensions is crucial
- **PIL integration**: Converting between torch.Tensor and PIL.Image

### Images (IMAGE)

#### Basic Properties
```python
# IMAGE tensor properties
image_tensor.shape  # [B, H, W, C] where C=3 for RGB
image_tensor.dtype   # torch.float32
image_tensor.range   # 0.0 to 1.0
```

#### Working with PIL.Image
```python
from PIL import Image, ImageOps
import torch

# Convert PIL.Image to torch.Tensor
def pil_to_tensor(pil_image):
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to tensor
    tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
    return tensor.unsqueeze(0)  # Add batch dimension

# Convert torch.Tensor to PIL.Image
def tensor_to_pil(tensor):
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and scale to 0-255
    array = (tensor * 255.0).clamp(0, 255).byte().numpy()
    return Image.fromarray(array)

# Example usage
def process_image(self, image_tensor):
    # Convert to PIL for processing
    pil_image = tensor_to_pil(image_tensor)
    
    # Process with PIL
    processed = ImageOps.invert(pil_image)
    
    # Convert back to tensor
    result = pil_to_tensor(processed)
    return (result,)
```

#### Shape Considerations
```python
# IMAGE is channel-last: [B, H, W, C]
image_tensor = torch.randn(1, 512, 512, 3)  # Batch, Height, Width, Channels

# Some operations expect channel-first: [B, C, H, W]
# Use permute to convert
channel_first = image_tensor.permute(0, 3, 1, 2)  # [B, C, H, W]
channel_last = channel_first.permute(0, 2, 3, 1)  # Back to [B, H, W, C]
```

### Masks (MASK)

#### Basic Properties
```python
# MASK tensor properties
mask_tensor.shape  # [B, H, W] or [H, W] (no channel dimension)
mask_tensor.dtype  # torch.float32
mask_tensor.range  # 0.0 to 1.0 (binary or continuous)
```

#### Understanding Mask Shapes
```python
def process_mask(self, mask):
    # Check mask shape
    if len(mask.shape) == 2:
        # [H, W] - single mask, add batch dimension
        mask = mask.unsqueeze(0)  # [1, H, W]
    elif len(mask.shape) == 3:
        # [B, H, W] - batch of masks
        pass
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    
    # Add channel dimension for operations
    mask_with_channels = mask.unsqueeze(-1)  # [B, H, W, 1]
    
    return mask_with_channels
```

#### Mask Operations
```python
def apply_mask(self, image, mask):
    # Ensure compatible shapes
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
    
    # Apply mask to image
    masked_image = image * mask
    return (masked_image,)
```

#### LoadImage Node Masks
```python
# LoadImage creates masks from alpha channel
# - Alpha values normalized to [0,1]
# - Values inverted (1 - alpha)
# - Default mask [1, 64, 64] for images without alpha
```

### Latents (LATENT)

#### Basic Properties
```python
# LATENT is a dictionary
latent_dict = {
    'samples': torch.Tensor,  # [B, C, H, W] where C=4
    # Other keys may include masks, etc.
}

# Access the latent tensor
latent_tensor = latent_dict['samples']
latent_tensor.shape  # [B, C, H, W] where C=4
```

#### Working with Latents
```python
def process_latent(self, latent):
    # Extract the samples tensor
    samples = latent['samples']  # [B, C, H, W]
    
    # Process the latent
    processed_samples = samples * 0.5  # Example operation
    
    # Return new latent dict
    return ({
        'samples': processed_samples
    },)
```

#### Latent vs Image Shapes
```python
# LATENT: channel-first [B, C, H, W]
latent_tensor = torch.randn(1, 4, 64, 64)  # Batch, Channels, Height, Width

# IMAGE: channel-last [B, H, W, C]
image_tensor = torch.randn(1, 512, 512, 3)  # Batch, Height, Width, Channels

# Convert between formats
def latent_to_image_shape(latent):
    # [B, C, H, W] -> [B, H, W, C]
    return latent.permute(0, 2, 3, 1)

def image_to_latent_shape(image):
    # [B, H, W, C] -> [B, C, H, W]
    return image.permute(0, 3, 1, 2)
```

### Common Operations

#### Tensor Manipulation
```python
def manipulate_tensor(self, tensor):
    # Add dimensions
    tensor = tensor.unsqueeze(0)    # Add batch dimension
    tensor = tensor.unsqueeze(-1)  # Add channel dimension
    
    # Remove dimensions
    tensor = tensor.squeeze(0)     # Remove batch dimension
    tensor = tensor.squeeze(-1)    # Remove channel dimension
    
    # Permute dimensions
    tensor = tensor.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
    
    return (tensor,)
```

#### Shape Matching
```python
def match_shapes(self, tensor1, tensor2):
    # Ensure same batch size
    if tensor1.shape[0] != tensor2.shape[0]:
        # Broadcast or repeat as needed
        pass
    
    # Ensure compatible dimensions
    if len(tensor1.shape) != len(tensor2.shape):
        # Add or remove dimensions
        pass
    
    return (tensor1, tensor2)
```

#### Memory Management
```python
def efficient_processing(self, tensor):
    # Use torch.no_grad() for inference
    with torch.no_grad():
        result = self.model(tensor)
    
    # Clear cache if needed
    torch.cuda.empty_cache()
    
    return (result,)
```

### Best Practices

#### Shape Validation
```python
def validate_tensor_shape(self, tensor, expected_shape):
    if tensor.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {tensor.shape}")
    return True
```

#### Type Safety
```python
def safe_tensor_operation(self, tensor):
    # Check tensor properties
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be torch.Tensor")
    
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    
    return tensor
```

#### Performance Tips
```python
def optimize_tensor_operations(self, tensor):
    # Use in-place operations when possible
    tensor.mul_(0.5)  # In-place multiplication
    
    # Use appropriate data types
    tensor = tensor.half()  # Use float16 if precision allows
    
    # Use efficient operations
    result = torch.nn.functional.relu(tensor)  # More efficient than manual ReLU
    
    return (result,)
```

### Common Pitfalls

#### Shape Mismatches
```python
# ❌ Wrong - shape mismatch
image = torch.randn(1, 512, 512, 3)  # [B, H, W, C]
mask = torch.randn(1, 512, 512)     # [B, H, W]
result = image * mask  # Error: shapes don't match

# ✅ Correct - match shapes
image = torch.randn(1, 512, 512, 3)  # [B, H, W, C]
mask = torch.randn(1, 512, 512, 1)   # [B, H, W, 1]
result = image * mask  # Works: [B, H, W, C] * [B, H, W, 1]
```

#### Dimension Confusion
```python
# ❌ Wrong - confusing dimensions
tensor = torch.randn(1, 4, 64, 64)  # Is this [B, C, H, W] or [B, H, W, C]?

# ✅ Correct - be explicit
latent = torch.randn(1, 4, 64, 64)  # [B, C, H, W] for latent
image = torch.randn(1, 64, 64, 3)  # [B, H, W, C] for image
```

#### Return Format
```python
# ❌ Wrong - missing trailing comma
def process(self, input):
    return (output)  # This is a tuple with one element, not a tuple

# ✅ Correct - proper tuple
def process(self, input):
    return (output,)  # This is a tuple with one element
```

## Node Lifecycle - How ComfyUI Loads Custom Nodes

### Loading Process
1. **ComfyUI starts** → Scans `custom_nodes` directory
2. **Finds Python modules** → Directories with `__init__.py`
3. **Attempts import** → Executes `__init__.py`
4. **Checks exports** → Looks for `NODE_CLASS_MAPPINGS`
5. **Loads nodes** → Makes them available in ComfyUI
6. **Error handling** → Reports failures but continues

### Module Structure
```
custom_nodes/
└── my_custom_node/          # Python module (directory)
    ├── __init__.py         # Required - module entry point
    ├── my_node.py          # Node implementations
    ├── requirements.txt    # Dependencies
    └── web/                # Client-side code (optional)
        └── js/
            └── my_extension.js
```

### __init__.py - Module Entry Point

#### Basic Structure
```python
from .my_node import MyCustomNode

NODE_CLASS_MAPPINGS = {
    "My Custom Node": MyCustomNode,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
```

#### Complete Example
```python
from .my_node import MyCustomNode, AnotherNode

NODE_CLASS_MAPPINGS = {
    "My Custom Node": MyCustomNode,
    "Another Node": AnotherNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "My Custom Node": "🎯 My Custom Node",
    "Another Node": "📊 Another Node",
}

WEB_DIRECTORY = "./web/js"

__all__ = [
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS',
    'WEB_DIRECTORY'
]
```

### Required Exports

#### NODE_CLASS_MAPPINGS
- **Purpose**: Maps node names to classes
- **Format**: `{"Node Name": NodeClass}`
- **Uniqueness**: Names must be unique across entire ComfyUI install
- **Required**: Must be present for module to be recognized

#### NODE_DISPLAY_NAME_MAPPINGS (Optional)
- **Purpose**: Human-readable display names
- **Format**: `{"Node Name": "Display Name"}`
- **Default**: Uses NODE_CLASS_MAPPINGS keys if omitted
- **Emoji support**: `"🎯 My Node"` for visual distinction

#### WEB_DIRECTORY (Optional)
- **Purpose**: Path to client-side JavaScript files
- **Format**: `"./web/js"` (relative to module)
- **File types**: Only `.js` files served
- **Convention**: Use `js` subdirectory

### Error Handling

#### Import Failures
- **ComfyUI continues**: Other modules still load
- **Console output**: Check Python console for errors
- **Module status**: Failed modules reported in logs

#### Common Issues
```python
# ❌ Wrong - missing __init__.py
custom_nodes/my_node.py

# ✅ Correct - proper module structure
custom_nodes/my_node/
├── __init__.py
└── my_node.py
```

### Development Workflow

#### 1. Create Module Structure
```bash
mkdir custom_nodes/my_custom_node
touch custom_nodes/my_custom_node/__init__.py
```

#### 2. Implement Nodes
```python
# my_custom_node/my_node.py
class MyCustomNode:
    # ... node implementation
```

#### 3. Register in __init__.py
```python
# my_custom_node/__init__.py
from .my_node import MyCustomNode

NODE_CLASS_MAPPINGS = {
    "My Custom Node": MyCustomNode,
}
```

#### 4. Test Loading
- **Restart ComfyUI** after changes
- **Check console** for import errors
- **Verify nodes** appear in Add Node menu

### Advanced Module Structure

#### Multiple Node Files
```python
# __init__.py
from .image_nodes import ImageProcessor, ImageFilter
from .text_nodes import TextProcessor, TextAnalyzer

NODE_CLASS_MAPPINGS = {
    "Image Processor": ImageProcessor,
    "Image Filter": ImageFilter,
    "Text Processor": TextProcessor,
    "Text Analyzer": TextAnalyzer,
}
```

#### Client-Side Integration
```python
# __init__.py
from .my_node import MyCustomNode

NODE_CLASS_MAPPINGS = {
    "My Custom Node": MyCustomNode,
}

WEB_DIRECTORY = "./web/js"
__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']
```

### Best Practices

#### Module Organization
- **Single responsibility**: One module per feature set
- **Clear naming**: Descriptive module and node names
- **Proper structure**: Follow ComfyUI conventions

#### Error Prevention
- **Test imports**: Verify modules load correctly
- **Handle exceptions**: Graceful error handling
- **Console monitoring**: Check for import errors

#### Performance
- **Lazy loading**: Import only when needed
- **Minimal dependencies**: Reduce startup time
- **Efficient code**: Optimize node execution

## Node Registration

### __init__.py
```python
from .my_node import MyCustomNode

NODE_CLASS_MAPPINGS = {
    "MyCustomNode": MyCustomNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyCustomNode": "🎯 My Custom Node",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

## Advanced Features

### IS_CHANGED Method
```python
@classmethod
def IS_CHANGED(cls, **kwargs):
    return float("NaN")  # Always update
    # or return hash of relevant inputs
```

### VALIDATE_INPUTS Method
```python
@classmethod
def VALIDATE_INPUTS(cls, **kwargs):
    # Validate inputs before processing
    return True
```

### Custom JavaScript
```python
@classmethod
def get_js(cls):
    js = """
    // Custom JavaScript code
    function myFunction() {
        // Implementation
    }
    """
    return js
```

## File Structure

```
my_custom_node/
├── __init__.py              # Node registration
├── my_node.py              # Node implementation
├── requirements.txt         # Dependencies
├── README.md               # Documentation
└── web/                    # Client-side assets (optional)
    └── my_extension.js
```

## Best Practices

### 1. Error Handling
```python
try:
    result = process_input(input_data)
except Exception as e:
    print(f"Error in MyCustomNode: {e}")
    return (None,)
```

### 2. Input Validation
```python
def process(self, image, strength):
    if image is None:
        raise ValueError("Image input is required")
    if not 0 <= strength <= 1:
        raise ValueError("Strength must be between 0 and 1")
```

### 3. Memory Management
```python
# Use torch.no_grad() for inference
with torch.no_grad():
    result = model(input_tensor)
```

### 4. Documentation
```python
class MyCustomNode:
    """
    My Custom Node - Does something useful
    
    Inputs:
        image: Input image tensor
        strength: Processing strength (0-1)
    
    Outputs:
        result: Processed image tensor
    """
```

## Common Patterns

### Image Processing Node
```python
class ImageProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Image Processing"
    
    def process(self, image, strength):
        # Process image
        processed = image * strength
        return (processed,)
```

### Model Wrapper Node
```python
class ModelWrapper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "input_data": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "Model"
    
    def run(self, model, input_data):
        with torch.no_grad():
            output = model(input_data)
        return (output,)
```

## Debugging

### Console Output
```python
print(f"Debug: Processing input shape {input_tensor.shape}")
```

### Error Messages
```python
try:
    result = process(input)
except Exception as e:
    print(f"Error in {self.__class__.__name__}: {e}")
    import traceback
    traceback.print_exc()
```

## Testing

### Manual Testing
1. Add node to workflow
2. Connect inputs
3. Execute workflow
4. Check outputs

### Unit Testing
```python
def test_my_node():
    node = MyCustomNode()
    result = node.process(test_input)
    assert result[0].shape == expected_shape
```

## Distribution

### Git Repository
- Clear README with installation instructions
- Proper versioning
- Example workflows

### Package Structure
- Follow ComfyUI conventions
- Include requirements.txt
- Document dependencies

## Step-by-Step Tutorial: Image Selector Node

### Prerequisites
- Working ComfyUI installation (manual installation recommended for development)
- Working comfy-cli installation

### Setting Up
```bash
cd ComfyUI/custom_nodes
comfy node scaffold
```

Answer the scaffold questions:
- full_name: Your name
- email: your@email.com
- github_username: your_github_username
- project_name: Your Project Name
- project_slug: your-project-slug
- project_short_description: Description of your nodes
- version: 0.0.1
- license: Choose appropriate license
- include_web_directory_for_custom_javascript: y/n

### Basic Node Structure

```python
class ImageSelector:
    CATEGORY = "example"
    
    @classmethod    
    def INPUT_TYPES(s):
        return { "required": { "images": ("IMAGE",) } }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "choose_image"
    
    def choose_image(self, images):
        brightness = list(torch.mean(image.flatten()).item() for image in images)
        brightest = brightness.index(max(brightness))
        result = images[brightest].unsqueeze(0)
        return (result,)
```

### Key Points
- **IMAGE** means image batch (single image = batch of size 1)
- Shape: `[B,H,W,C]` where B=batch, H=height, W=width, C=channels
- `unsqueeze(0)` adds batch dimension
- Return tuple: `(result,)` - trailing comma essential

### Register the Node

```python
NODE_CLASS_MAPPINGS = {
    "Example": Example,
    "Image Selector": ImageSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Example Node",
    "Image Selector": "Image Selector",
}
```

### Adding Options

```python
@classmethod    
def INPUT_TYPES(s):
    return { 
        "required": { 
            "images": ("IMAGE",), 
            "mode": (["brightest", "reddest", "greenest", "bluest"],)
        } 
    }

def choose_image(self, images, mode):
    batch_size = images.shape[0]
    brightness = list(torch.mean(image.flatten()).item() for image in images)
    
    if (mode=="brightest"):
        scores = brightness
    else:
        channel = 0 if mode=="reddest" else (1 if mode=="greenest" else 2)
        absolute = list(torch.mean(image[:,:,channel].flatten()).item() for image in images)
        scores = list(absolute[i]/(brightness[i]+1e-8) for i in range(batch_size))
    
    best = scores.index(max(scores))
    result = images[best].unsqueeze(0)
    return (result,)
```

### Server-Client Communication

#### Send Message from Server
```python
from server import PromptServer

def choose_image(self, images, mode):
    # ... processing code ...
    
    PromptServer.instance.send_sync("example.imageselector.textmessage", {
        "message": f"Picked image {best+1}"
    })
    return (result,)
```

#### Client Extension
Create `web/js/imageSelector.js`:
```javascript
app.registerExtension({
    name: "example.imageselector",
    async setup() {
        function messageHandler(event) { 
            alert(event.detail.message); 
        }
        app.api.addEventListener("example.imageselector.textmessage", messageHandler);
    },
})
```

#### Update __init__.py
```python
WEB_DIRECTORY = "./web/js"
__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']
```

### Development Workflow
1. **Make changes** to your node code
2. **Restart ComfyUI** server
3. **Reload webpage** in browser
4. **Test workflow** with your node

### Complete Example Structure
```
my_custom_node/
├── __init__.py              # Node registration + WEB_DIRECTORY
├── src/
│   └── nodes.py            # Node implementations
├── web/
│   └── js/
│       └── imageSelector.js # Client-side JavaScript
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## Resources

### Official Documentation
- ComfyUI GitHub: https://github.com/comfyanonymous/ComfyUI
- Custom Nodes Guide: https://github.com/comfyanonymous/ComfyUI/wiki

### Example Repositories
- cookiecutter-comfy-extension
- ComfyUI-React-Extension-Template
- ComfyUI_frontend_vue_basic

### Community
- ComfyUI Discord
- GitHub Issues
- Community Forums

---

*This guide is based on ComfyUI's official documentation and community best practices.*
