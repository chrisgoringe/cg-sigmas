from comfy_api.latest import ComfyExtension, io
from .sigmas import (
    EmptyModel, ManualSigmas, KL_Optimal,                        
    SplitSigmasAtSigmaValue,ConcatentateSigmas, ChangeStepCount,
    GraphSigmas, DisplayAnything
)


async def comfy_entrypoint() -> ComfyExtension:
    class CGSigmasExtension(ComfyExtension):
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [
                EmptyModel, ManualSigmas, KL_Optimal,                        # tools for getting sigmas
                SplitSigmasAtSigmaValue, ConcatentateSigmas, ChangeStepCount,
                GraphSigmas, DisplayAnything
            ]
        
    return CGSigmasExtension()

WEB_DIRECTORY = "./js"
__all__ = [ "WEB_DIRECTORY", ]