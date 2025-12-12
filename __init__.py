from comfy_api.latest import ComfyExtension, io
from .sigmas import (
    EmptyModel, ManualSigmas, KL_Optimal, Flux2Sigmas,          
    SplitSigmasAtSigmaValue,ConcatentateSigmas, ChangeStepCount,
    GraphSigmas, DisplayAnything
)
from .progress_sampler import ProgressSampler
from .shift import ExploreShift
from .sampling import UpstepSamplerWrap, SamplerSwitcherWrap

async def comfy_entrypoint() -> ComfyExtension:
    class CGSigmasExtension(ComfyExtension):
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [
                EmptyModel, ManualSigmas, KL_Optimal, Flux2Sigmas,                    
                SplitSigmasAtSigmaValue, ConcatentateSigmas, ChangeStepCount, 
                GraphSigmas, DisplayAnything,
                ProgressSampler,
                ExploreShift,
                UpstepSamplerWrap, SamplerSwitcherWrap
            ]
        
    return CGSigmasExtension()

WEB_DIRECTORY = "./js"
__all__ = [ "WEB_DIRECTORY", ]