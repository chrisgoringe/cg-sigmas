import comfy.supported_models
from comfy.supported_models_base import BASE
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io, ui
import folder_paths

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from typing import Optional
import random, os, math

from comfy.comfy_types.node_typing import IO
anything = io.Custom(IO.ANY)

CSC_DESCRIPTION = '''Change the number of steps in a sigmas list by interpolating. 
Multiplication is applied first, then addition, so
steps_after = (steps_before) * multiplier + adder
'''

def interpolate(x:float, ys:list[float]) -> float:
    if x==0: return ys[0 ]
    if x==1: return ys[-1]

    x_prime = (len(ys)-1) * x
    i, f = int(x_prime), x_prime%1

    return (1-f)*ys[i] + f*ys[i+1]

class KL_Optimal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "KL_Optimal",
            category = "quicknodes/sigmas",
            inputs  = [
                io.Int.Input("steps", default=20, min=1, max=1000),
                io.Float.Input("sigma_max", default=1.0, min=0.0),
                io.Float.Input("sigma_min", default=0.0, min=0.0)
            ],
            outputs = [
                io.Sigmas.Output("sigmas", display_name="sigmas"),
            ]
        )
    
    @classmethod
    def execute(cls, steps:int, sigma_max:float=1.0, sigma_min:float=0.0) -> io.NodeOutput: # type: ignore
        n = steps + 1
        fraction = torch.arange(n, dtype=torch.float).div_(n - 1)  # n equally spaced values from 0.0 to 1.0
        sigmas = ( (1 - fraction) * math.atan(sigma_max) + fraction * math.atan(sigma_min) ).tan_()
        return io.NodeOutput( sigmas, )
    
class DisplayAnything(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "DisplayAnything",
            category = "quicknodes/sigmas",
            inputs   = [ anything.Input("anything", optional=True), ],
            outputs  = [ io.String.Output("string", display_name="string"), ],
            is_output_node = True
        )
    
    @classmethod
    def execute(cls, anything) -> io.NodeOutput: # type: ignore
        s:str = str(anything)
        return io.NodeOutput( s, ui = {"cg_sigmas_display_text":s} )

class ChangeStepCount(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "ChangeStepCount",
            category = "quicknodes/sigmas",
            description = CSC_DESCRIPTION,
            inputs  = [
                io.Sigmas.Input("sigmas"),
                io.Float.Input("multiplier", default=1.0, tooltip="Multiply the number of steps by this"),
                io.Int.Input("adder", default=0, min=-100, max=100, tooltip="Add (or subtract) this number of steps")
            ],
            outputs = [
                io.Sigmas.Output("sigmas_out", display_name="sigmas"),
            ] 
        )
    
    @classmethod
    def execute(cls, sigmas:torch.Tensor|list[float], multiplier:float, adder:int) -> io.NodeOutput: # type: ignore
        if len(sigmas)==0: return io.NodeOutput( sigmas, )

        sigmas          = [float(s) for s in sigmas]
        old_step_count  = len(sigmas)-1
        new_step_count  = round(old_step_count * multiplier) + adder

        new_sigmas = [interpolate(i/new_step_count, sigmas) for i in range(new_step_count+1)]
        return io.NodeOutput( torch.tensor(new_sigmas), )

class ConcatentateSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "ConcatentateSigmas",
            category = "quicknodes/sigmas",
            inputs   = [ io.Sigmas.Input("sigmas1"), io.Sigmas.Input("sigmas2"), ],
            outputs  = [ io.Sigmas.Output("sigmas", display_name="sigmas"), ] 
        )
    
    @classmethod
    def execute(cls, sigmas1:torch.Tensor|list[float], sigmas2:torch.Tensor|list[float]) -> io.NodeOutput: # type: ignore
        longlist = [ float(i) for i in sigmas1 ] + [ float(i) for i in sigmas2[1:] ]
        return io.NodeOutput( torch.tensor(longlist), )

class SplitSigmasAtSigmaValue(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "SplitSigmasAtSigmaValue",
            category = "quicknodes/sigmas",
            inputs   = [
                io.Sigmas.Input("sigmas"),
                io.Float.Input("split_at", default=0.875, tooltip="For Wan 2.2: T2V 0.875, I2V 0.9"),
                io.Boolean.Input("adjust", default=False, tooltip="If true, adjust the sigma value at the cut point"),
            ],
            outputs  = [
                io.Sigmas.Output("high", display_name="high sigmas"),
                io.Sigmas.Output("low", display_name="low sigmas"),
                io.Int.Output("split_index", tooltip="The step at which the sigmas were cut"),
            ] 
        )
    
    @classmethod
    def execute(cls, sigmas:list[float], split_at:float, adjust:float) -> io.NodeOutput: # type: ignore
        err = [ abs(s-split_at) for s in sigmas ]
        idx = err.index(min(err))
        if adjust: sigmas[idx] = split_at 
        return io.NodeOutput( sigmas[:idx+1], sigmas[idx:], idx )

class ManualSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "ManualSigmas",
            category = "quicknodes/sigmas",
            inputs   = [
                io.String.Input("sigmas", multiline=True, tooltip="Comma separated list of values. Starting value depends on model (SD 1.5 and SDXL 14.615, WAN 1.0), end with 0.0. Whitespace will be trimmed."),
                io.Int.Input("steps", optional=True, default=0, tooltip="If set to a positive integer, interpolate to that number of steps.")
            ],
            outputs  = [
                io.Sigmas.Output("sigmas_out", display_name="sigmas"),
                io.String.Output("warnings", display_name="warnings"),
            ] 
        )
    
    @classmethod
    def execute(cls, sigmas:str, steps:int) -> io.NodeOutput: # type: ignore
        s_list = [s.strip() for s in sigmas.split(",")]
        s_floats = []
        warnings = []
        for s in s_list:
            try: s_floats.append( float(s) )
            except: warnings.append(f"Failed to parse {s} as a float")
        if len(s_floats)<2: warnings.append(f"Fewer than two values")
        if s_floats[-1]!=0.0: warnings.append(f"Doesn't end at 0.0")

        for i in range(0, len(s_floats)-2):
            if s_floats[i]<=s_floats[i+1]: warnings.append(f"Sigmas increase from {s_floats[i]} to {s_floats[i+1]}")

        if steps:
            if len(s_floats)>1:
                s_floats = [interpolate(x = i/steps, ys = s_floats) for i in range(steps+1)]
            else:
                warnings.append(f"Need at least two points to interpolate")

        return io.NodeOutput( torch.Tensor(s_floats), "\n".join(warnings) ) 


def solve(y_target:float, ys:list[float]) -> float:
    x = 0
    for i, y in enumerate(ys):
        if y < y_target and x==0: x = i-1
    dy_by_dx = ys[x+1] - ys[x]  
    dy       = y_target - ys[x] 
    dx = dy / dy_by_dx
    return x + dx

class GraphSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "GraphSigmas",
            category = "quicknodes/sigmas",
            inputs   = [
                io.Sigmas.Input("sigmas"),
                io.String.Input("title", optional=True),
                io.String.Input("labels", optional=True),
                io.Float.Input("high_low_divide", optional=True, default=0.0, tooltip="Add a horizontal line at this value of sigma"),
                io.Boolean.Input("show_intercepts", default=False)
            ],
            outputs = [
                io.Image.Output("graph", display_name="graph"),
            ],
            is_output_node = True,
            is_input_list = True,
        )
    
    @classmethod
    def execute(cls, sigmas:list[torch.Tensor|list[float]], title:Optional[list[str]]=None, labels:Optional[list[str]]=None, high_low_divide:Optional[list[float]]=None, show_intercepts:list[bool]=[False,]) -> io.NodeOutput: # type: ignore

        sigmas_lists = [[float(s) for s in sigma_list] for sigma_list in sigmas] 
        amarker = high_low_divide[0] if high_low_divide else None
        fig, ax = plt.subplots()

        labels = [l.strip() for l in labels if l.strip()] if labels else []
        text_labels = labels if len(labels)==len(sigmas_lists) else \
                                ([f"#{i+1}" for i in range(len(sigmas_lists))] if len(sigmas_lists)>1 else None)

        if len(show_intercepts)==1 and len(sigmas_lists)>1:
            show_intercepts = [show_intercepts[0] for _ in sigmas_lists]

        if title and title[0]: ax.set_title(title[0])
        ax.set_xlabel("steps")
        ax.set_ylabel("sigma")
        
        for i, sigma_list in enumerate(sigmas_lists):
            if text_labels: ax.plot(range(len(sigma_list)), sigma_list, label=text_labels[i]) 
            else: ax.plot(range(len(sigma_list)), sigma_list)

        if amarker:
            ax.plot([0,len(sigmas_lists[0])-1], [amarker, amarker], label=f"sigma = {amarker}")
            for i, sigma_list in enumerate(sigmas_lists):
                if show_intercepts[i]:
                    mx = solve(amarker, sigma_list)
                    ax.plot([mx, mx], [min(sigma_list), max(sigma_list)], label=f"step = {mx:>.1f}")

        ax.legend()

        filename = f"{random.randint(1000000,9999999)}.png"
        filepath = os.path.join(folder_paths.get_temp_directory(), filename)
        fig.savefig(filepath)
        plt.close(fig)

        with Image.open(filepath) as img:
            img = img.convert("RGB")
            image:torch.Tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

        return io.NodeOutput(image, ui=ui.SavedImages([ui.SavedResult(filename, "", io.FolderType.temp)]))

class EmptyModel(io.ComfyNode):
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "EmptyModel",
            category = "quicknodes/sigmas",
            inputs = [
                io.Combo.Input("model_type", [k for k in cls.available_model_types()])
            ],
            outputs = [
                io.Model.Output("model")
            ],
        )

    @classmethod
    def execute(cls, model_type:str) -> io.NodeOutput: # type: ignore
        model_config_class, extra = cls.available_model_types()[model_type]
        unet_config = getattr(model_config_class, 'unet_config')
        for k in [k for k in extra if k not in unet_config]: unet_config[k] = extra[k]
        model_config = model_config_class(unet_config)
        model = model_config.get_model(state_dict={})
        wrapped = ModelPatcher(model,'meta','meta')
        return io.NodeOutput(wrapped)

    @classmethod
    def available_model_types(cls) -> dict[str,tuple[type[BASE],dict]]:
        return {
            "WAN" : (comfy.supported_models.WAN22_T2V, {})
        }
    
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return random.random()