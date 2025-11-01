from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustom

sampler_custom = SamplerCustom()

class ProgressSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "ProgressSampler",
            category = "quicknodes/sigmas",
            description = "Returns lists of latents (with noise and denoised) after each step",
            inputs  = [
                io.Model.Input("model"),
                io.Int.Input("noise_seed", min=0, max=0xffffffffffffffff),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent"),
            ],
            outputs = [
                io.Latent.Output("latents", display_name="raw latents", is_output_list=True),
                io.Latent.Output("denoised_latents", display_name="denoised latents", is_output_list=True),
            ]
        )
    
    @classmethod
    def execute(cls, sigmas, latent, **kwargs): # type: ignore
        latents          = []
        denoised_latents = []

        for step in range(len(sigmas)-1):
            latent, denoised_latent = sampler_custom.sample(
                add_noise    = (step==0), 
                sigmas       = sigmas[step:step+2],
                latent_image = latent,
                **kwargs
            )
            latents.append(latent)
            denoised_latents.append(denoised_latent)

        return io.NodeOutput(latents, denoised_latents)