import torch
from comfy.samplers import KSAMPLER
from comfy_api.latest import io
from typing import Callable
from .sigmas import split_sigmas_at

# 
def pythagorean(a,b): 
    return (b ** 2 - a ** 2) ** 0.5 

def linear(a,b): 
    return b - a

MODES:dict[str, Callable[[float,float], float]] = {
    "pythagorean" : pythagorean,
    "linear" : linear
}

def wrap_sampling_for_upstep(wrapped:Callable, mode:str, scale:float):
    def wrapped_function(model, x, sigmas, **kwargs):
        upsteps   = [i for i in range(len(sigmas) - 1) if sigmas[i+1]  > sigmas[i]]
        next_step = 0
        for upstep in upsteps:
            # take any steps before the upstep all in one call
            if upstep > next_step: 
                x = wrapped(model, x, sigmas[next_step:upstep+1], **kwargs)
                next_step = upstep

            # take the upstep
            eps = torch.randn_like(x) * kwargs.get('s_noise', 1.0)
            x = x + scale * eps * MODES[mode](sigmas[next_step], sigmas[next_step+1])      
            next_step += 1

        # take the remaining steps
        x = wrapped(model, x, sigmas[next_step:], **kwargs)
        return x
    
    return wrapped_function

def wrap_sampling_for_switch(high:Callable, low:Callable, threshold:float, high_options, low_options):
    def wrapped_function(model, x, sigmas, **kwargs):
        high_sigmas, low_sigmas, _ = split_sigmas_at(sigmas, threshold)
        x = high(model, x, high_sigmas, **high_options, **kwargs)
        x = low(model, x, low_sigmas, **low_options, **kwargs)
        return x
    
    return wrapped_function

class UpstepSamplerWrap(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "UpstepSamplerWrap",
            category = "quicknodes/sigmas",
            inputs   = [ 
                io.Sampler.Input("sampler"),
                io.Combo.Input("mode", options=[m for m in MODES]),
                io.Float.Input("scale", default=1.0)
            ],
            outputs = [ io.Sampler.Output("wrapped", display_name="sampler"), ],
        )
    
    @classmethod
    def execute(cls, sampler, mode, scale) -> io.NodeOutput: # type: ignore
        wrapped = KSAMPLER( wrap_sampling_for_upstep(sampler.sampler_function, mode, scale), sampler.extra_options, sampler.inpaint_options  )
        return io.NodeOutput(wrapped,)

class SamplerSwitcherWrap(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "SamplerSwitcherWrap",
            category = "quicknodes/sigmas",
            inputs   = [ 
                io.Sampler.Input("high_noise_sampler"),
                io.Sampler.Input("low_noise_sampler"),
                io.Float.Input("threshold", default=0.4, step=0.01, tooltip="Value of sigma at which to switch from high to low")
            ],
            outputs = [ io.Sampler.Output("wrapped", display_name="sampler"), ],
        )
    
    @classmethod
    def execute(cls, high_noise_sampler:KSAMPLER, low_noise_sampler:KSAMPLER, threshold:float) -> io.NodeOutput: # type: ignore
        wrapped = KSAMPLER( 
            wrap_sampling_for_switch(
                high         = high_noise_sampler.sampler_function, 
                low          = low_noise_sampler.sampler_function, 
                threshold    = threshold,
                high_options = high_noise_sampler.extra_options, 
                low_options  = low_noise_sampler.extra_options
            ), {}, high_noise_sampler.inpaint_options  )
        return io.NodeOutput(wrapped,)