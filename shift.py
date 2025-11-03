from comfy_api.latest import io, ui
import matplotlib.pyplot as plt

import math

from .modules.utils import safe_tempfile, load_image, label_plot

class ExploreShift(io.ComfyNode):
    FLOW_TYPES = ['Discrete Flow', 'Flux Flow']
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "ExploreShift",
            category = "quicknodes/sigmas",
            inputs   = [
                io.Combo.Input("flowtype", options=cls.FLOW_TYPES),
                io.String.Input("shifts", default="8.0", tooltip="Comma separated list of values"),
            ],
            outputs = [
                io.Image.Output("graph", display_name="graph"),
            ],
            is_output_node = True,
        )
    
    @classmethod
    def execute(cls, flowtype:str, shifts:str) -> io.NodeOutput: # type: ignore
        eps = 1e-8
        x = [ t*0.0001 for t in range(0,10001) ] 
        one_minus_t = [ (1-t) for t in x ]

        ss = []
        for s in shifts.split(","):
            try:
                ss.append(float(s.strip()))
            except Exception as e:
                print(f"{e} when handling {s} in {shifts}, which should be a comma separated list of values")

        fig, ax = plt.subplots()

        for shift in ss:
            if flowtype == cls.FLOW_TYPES[0]:
                y = [t * (shift / (1 + (shift - 1) * t + eps)) for t in x]
            elif flowtype == cls.FLOW_TYPES[1]:
                y = [ math.exp(shift) / ( math.exp(shift) + (1/(t+eps)) - 1 ) for t in x]
            ax.plot(one_minus_t, y, label=f"shift = {shift}") # pyright: ignore[reportPossiblyUnboundVariable]

        label_plot(ax, flowtype, xlabel="(1-t) (fraction of steps used)", xformat="{x:>4.2f}", ylabel="t' (fraction of 'time' left)", yformat="{x:>4.2f}")

        filepath = safe_tempfile()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)

        image = load_image(filepath)

        return io.NodeOutput(image, ui=ui.SavedImages([ui.SavedResult(filepath.name, "", io.FolderType.temp)]))