import comfy.supported_models
from comfy.supported_models_base import BASE
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io, ui
from comfy_extras.nodes_flux import Flux2Scheduler
import nodes

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


from typing import Optional
import random, math
from .modules.utils import safe_tempfile, load_image, label_plot

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
    
class Flux2Sigmas(Flux2Scheduler):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "Flux2Sigmas",
            category = "quicknodes/sigmas",
            inputs  = [
                io.Int.Input("steps", default=20, min=1, max=4096),
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs = [
                io.Sigmas.Output("sigmas", display_name="sigmas"),
            ]
        )

    
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


def solve_for_x(y_target:float, ys:list[float]) -> float:
    x = 0
    for i, y in enumerate(ys):
        if y < y_target and x==0: x = i-1
    dy_by_dx = ys[x+1] - ys[x]  
    dy       = y_target - ys[x] 
    dx = dy / dy_by_dx
    return x + dx

def solve_for_y(x_target:float, ys:list[float]) -> float:
    x = int(x_target)
    dy_by_dx = ys[x+1] - ys[x]  
    dx       = x_target - x      
    dy = dy_by_dx * dx
    return ys[x] + dy

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
                io.Int.Input("step_divide", optional=True, default=0, tooltip="If set to a positive integer, show vertical line at this step"),
                io.Boolean.Input("show_intercepts", default=False, tooltip="If true, show vertical lines where horizontal line intersects sigmas"),
            ],
            outputs = [
                io.Image.Output("graph", display_name="graph"),
            ],
            is_output_node = True,
            is_input_list = True,
        )
    
    @classmethod
    def execute(cls, # type: ignore
                sigmas:list[torch.Tensor|list[float]], 
                title:Optional[list[str]]=None, 
                labels:Optional[list[str]]=None, 
                high_low_divide:Optional[list[float]]=None, 
                step_divide:Optional[list[int]]=None,
                show_intercepts:list[bool]=[False,]
            ) -> io.NodeOutput: 
        
        # Process input lists
        n = len(sigmas)

        sigmas_lists = [[float(s) for s in sigma_list] for sigma_list in sigmas] 
        graph_title  = title[0] if title else None
        labels       = cls.extend_list(labels, n) 
        text_labels  = labels if labels[0] else ([f"#{i+1}" for i in range(n)] if n>1 else [None,])
        ymarker      = cls.extend_list(high_low_divide, n)
        xmarker      = cls.extend_list(step_divide, n)
        intercepts   = cls.extend_list(show_intercepts, n)

        min_x = 0
        max_x = max([len(sigma_list)-1 for sigma_list in sigmas_lists])
        min_y = min(s for sigma_list in sigmas_lists for s in sigma_list)
        max_y = max(s for sigma_list in sigmas_lists for s in sigma_list)

        fig, ax = plt.subplots()

        for sigma_list, label, x_mark, y_mark, intercept in zip(sigmas_lists, text_labels, xmarker, ymarker, intercepts):
            ax.plot(range(len(sigma_list)), sigma_list, label=label) 
            if y_mark and min_y < y_mark < max_y:
                ax.plot([min_x, max_x], [y_mark, y_mark], label=f"sigma = {y_mark}")
                if intercept:
                    xp = solve_for_x(y_mark, sigma_list)
                    ax.plot([xp, xp], [min_y, max_y], label=f"step = {xp:>.1f}")
            if x_mark and min_x < x_mark < max_x: 
                ax.plot([x_mark, x_mark], [min_y, max_y], label=f"step = {x_mark}")
                if intercept:
                    yp = solve_for_y(x_mark, sigma_list)
                    ax.plot([min_x, max_x], [yp, yp], label=f"sigma = {yp:>.2f}")

        label_plot( ax, graph_title, xlabel="steps", xformat="{x:.0f}", ylabel="sigma", yformat="{x:>5.2f}" )

        filepath = safe_tempfile()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)

        image = load_image(filepath)


        return io.NodeOutput(image, ui=ui.SavedImages([ui.SavedResult(filepath.name, "", io.FolderType.temp)]))
    
    @staticmethod
    def extend_list(lst:Optional[list], target_length:int) -> list:
        if not lst: lst = [None,]
        while len(lst)<target_length: lst.append(lst[-1])
        return lst

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