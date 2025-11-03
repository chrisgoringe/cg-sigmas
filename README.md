# Sigmas, Scheduling, Shift and Stuff

A collection of nodes I put together to try to understand sigmas and scheduling a bit better.

They can be found under `quicknodes/sigmas`.

## Version history

*Update - 3/11/2025*
- added discussion of shift and the ExploreShift node
- added discussion of 'lying sigmas'
- a few clarifications

*First version - 1/11/2025*

## What are sigmas anyway?

<details>
<summary>Sigma is a measure of how much noise there is assumed to be left in an image as you go through the denoising process. </summary>

`sigma=0.0` means that there is no noise. For the purposes of this explanation, we'll say that `sigma=1.0` is pure noise 
(some models, such as SDXL, use `sigma=14.1` for pure noise, for reasons that aren't clear to me, but the values just scale).

The scheduler creates a list of `sigmas` which tell the denoising process how quickly the noise should be removed.
A simple list might look like this: `[1.0, 0.5, 0.2, 0.0]`. This would tell the denoising process that we start
with `1.0` noise, the first step reduces it to `0.5`, the second to `0.2` and the third to `0.0`. 
Notice that the list of sigmas is one longer than the number of steps.

If you are doing img2img, `sigma` will start at the value you choose for `denoise`(and this amount of noise will be
added to the image at the start). Essentially you are just starting part way through the process.

In some sampler nodes you just pick a scheduler and leave it at that. 
The `SamplerCustom` and `SamplerCustomAdvanced` nodes allow you to specify the `sigmas` that you want to use.
</details>

## Why does this matter?

<details>
<summary>Different aspects of the image develop at different values of sigma</summary>

Imagine someone walking towards you on a foggy day - the fog is the noise, the person is the image.
As they approach, the noise gets less, allowing you to start to see them.

At first, when the noise is still quite high, you can only make out the rough shape of the figure.
As the noise gets less, you start to see major features - arms and legs, perhaps. 
As they get closer still, those major features don't change much, but more detail emerges.

You can see how this works using the `ProgressSampler` node. 
This is a replacement for the built in `SamplerCustom` node which outputs lists of latents (raw and denoised),
one entry after each step. If you connect it up where a `SamplerCustom` would normally go, 
and decode the denoised latents you will get a set of images showing the development of the image step by step.

![screenshot showing the use of the SamplerCustom node](images/progress.png)

The denoised latents are what you would get if the list of sigmas was truncated at that point with a zero -
meaning remove all the remaining noise this step. That's what we want for this purpose.

The raw latents are the latents with the noise left in, 
as if you truncated the list of sigmas without setting the end point to zero. 
That's what you want, if you are passing a partially denoised image on to a different model 
(a refiner, or the low noise model from a high/low pair as in WAN), or switching samplers,
changing cfg, adding or removing LoRAs, or whatever.

</details>

<details>
<summary>Images generated at each step in a 20 step denoise</summary>

|Step|||
|-|-|-|
|1,2|![1](images/progress/00.png)|![1](images/progress/01.png)|
|3,4|![1](images/progress/02.png)|![1](images/progress/03.png)|
|5,6|![1](images/progress/04.png)|![1](images/progress/05.png)|
|7,8|![1](images/progress/06.png)|![1](images/progress/07.png)|
|9,10|![1](images/progress/08.png)|![1](images/progress/09.png)|
|11,12|![1](images/progress/10.png)|![1](images/progress/11.png)|
|13,14|![1](images/progress/12.png)|![1](images/progress/13.png)|
|15,16|![1](images/progress/14.png)|![1](images/progress/15.png)|
|17,18|![1](images/progress/16.png)|![1](images/progress/17.png)|
|19,20|![1](images/progress/18.png)|![1](images/progress/19.png)|

</details>

## Key Point

*The broad features of an image are determined during the early steps, when `sigma` is still high.*

*The fine details are determined when `sigma` is low.*

## Implications

<details>

<summary>If you want to improve details, you want the denoising process to take more care (by taking more, smaller, steps)
when sigma is relatively small; if you are wanting to improve broad features, you want more steps while sigma is high.</summary>

That is exactly what `shift` does in a model like WAN (using the `ModelSamplingSD3` node):

![screenshot showing the decay of sigma with different shifts](images/shift.png)

Or the different beta schedulers - beta57 pays more attention to detail than beta (=beta66)

![screenshot showing the decay of sigma with different parameters](images/betas.png)

The nodes in this pack help to visualise the `sigmas` created by different schedulers, and also help you manipulate them.

</details>

---

# Visualising Sigmas

<details>
<summary>You can visualise the sigmas you are using with the `GraphSigmas` node. </summary>
Plug in a list of `sigmas`, and it plots a graph for you.
You can optionally add a y-intercept, and get it to give you the step number at which this intercept is crossed 
(useful for models like WAN which have high and low models designed for different ranges of `sigma`).

So here's a 20 step diffusion with WAN, using `beta57` and `shift=8.0`. The intercept is at `sigma=0.9` which
is the transition from the High to Low model in I2V. For T2V the transition is 0.875.

![screenshot showing the decay of sigma in a beta57 scheduler with shift of 8](images/graph0.png)

In the workflows folder [here](workflows) you can find the workflow from that screenshot (workflow0.json).
Have a play with the alpha and beta and shift values. You'll also see another node from this deck, `DisplayAnything`
which gives you the values of `sigma` in numeric form - this might be useful later.

</details>

# Generating Sigmas

<details>
<summary>Sigmas are determined by the scheduler, but you can refine or replace them</summary>

In the base Comfy nodes, under `sampling/custom_sampling/schedulers` are a load of different schedulers. They require 
a model to be connected (in most cases that is just used to determine the starting value of `sigma`). To speed things
up, you can use the `EmptyModel` node in this pack, which just pretends to be a model without loading anything.

Also in this pack you can find `KL_Optimal`, which is another scheduler, and `ManualSigmas`, which allows you to 
enter a series of numbers (and optionally set a number of steps - leave this at zero to get out the list you put in, 
or change it to get the node to interpolate).

One trick is to copy the values from `DisplayAnything` into `ManualSigmas`, and you can then tweak the numbers as you like.
`ManualSigmas` will ignore all whitespace, and just needs the values to be separated by commas. The 'warnings' output
can be plugged into a `DisplayAnything` node, and it will tell you if you didn't end at zero, or if you increased sigma.

![screenshot showing the use of the ManualSigmas node](images/manual.png)

</details>

# Manipulating Sigmas

## Splitting and Concatenating

<details>
<summary>Splitting lists of sigmas can be useful, and there are several ways to do it</summary>

The Comfy core nodes include two nodes for splitting sigmas - `SplitSigmas`, which splits them at a given step, 
and `SplitSigmasDenoise` which splits then a specified fraction of the way through. While scaling the number of
steps with the denoise fraction is common practice, I'd suggest you ought to use the sigma value...

For that reason, and to support changing model or sampler part way through, this pack adds a third
option, `SplitSigmasAtSigmaValue`, which finds the point nearest to a transition value (say, 0.9 for WAN) and
splits the sigmas there.

You can play with those three with workflow1.json [here](workflows).

Recombine the sigmas into a single list with `ConcatenateSigmas`.
</details>

## Stretching and Compressing

<details>
<summary>The ChangeStepCount node allows you to manipulate the sigma list</summary>

The `ChangeStepCount` node takes a list of sigmas, and allows you to change the number of steps by multiplying, 
adding, or both. The start and end values will be unchanged. 

This only really makes sense if you have split the sigma list (otherwise, just change the step count!).
But if you have, you might want to take extra time in the high
sigma space - on the right, we've tripled the number of high sigma steps:

![screenshot showing the use of the ChangeStepCount node](images/stretch.png)

You can see the effects of these nodes in workflow2.json [here](workflows).
</details>

---

# How does shift work?

<details>
<summary>Shift works by modifying the time axis.</summary> 

Imagine the denoising steps as a timer running down from `t=1` to `t=0`. If you do a twenty step denoise, 
after 7 steps `t = 13/20`, because there are 13 out of 20 steps left.

The model is actually given a different timestep, `t'`, determined by shift.

WAN (and other models using the `ModelSamplingSD3` node) uses 'Discrete Flow', in which
`t' = t * [shift / (1 + (shift - 1) * t)]`, which looks like this:

![discrete](images/discrete.png)

Note the x-axis is `1-t`, the fraction of steps taken, so it runs from left to right like the sigma plots do.
You'll also notice that for `shift=1.0` we have `t'=t`, an unshifted time axis.

You can create plots like this with the `ExploreShift` node in this pack. 

Flux uses shift differently. `t' = e^shift / ( e^shift + (1/t) - 1 )`. 

<details>
<summary>Firstly, shift depends on image size</summary>  
The `ModelSamplingFlux` node doesn't allow you to set the shift directly, because it sets a value of shift which
can depend upon the image area. The node specifies `base_shift`, which is used for an image
that is 256x256 pixels in area, and `max_shift`, which is used for an image that is 1024x1024. For sizes of image in between, 
the value of shift is linearly interpolated (based on the area); for small and larger images the same formula is used, so 
shift can be less than `base_shift` (for smaller images) or greater than the misnamed `max_shift` (for larger ones)
</details>

Secondly the formula used, `t' = e^shift / ( e^shift + (1/t) - 1 )` is 'unshifted' for `shift=0.0`, 
and requries smaller variations in value of shift.

![flux](images/fluxflow.png)

Differences notwithstanding, the general effect is similar (compare flux shift of 1.15 with discrete shift of 3).
Larger values of shift cause the curve to stay high for longer, which means the model acts as if it were early in
the denoising process for longer, as reflected in sigma remaining high for longer.

</details>

---

# Lying Sigmas - what, and why?

<details>
<summary>"Lying Sigmas" describes a process by which the sampler gives the model incorrect information 
about the noise left in the image.</summary>

Right at the heart of the image generation process is the model, which has been trained to separate image from noise.
When the model was trained it was given images with different amounts of noise, along with the corresponding 
`sigma` value. So it 'knows' what sort of noise it ought to predict for a given value of `sigma`.

In two or more part models, like WAN 2.2, the 'high' and 'low' models are trained on sources with different
ranges of noise - 'high' is trained to identify noise in high-noise/high-sigma images, 'low' in low-noise/low-sigma images.

In sampling, the sampler gives the model the noisy image, and the sigma value, so the model 'knows' how much and what sort 
of noise it is looking for. At its simplest, a "lying sigma" passes the model an inaccurate value of sigma, which tricks the
model into looking for the wrong amount of noise.

Here's the core of the code from [Detail Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon)'s LyingSigmaSampler:

```python
    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.max().detach().cpu())
        if end_sigma <= sigma_float <= start_sigma:
            sigma = sigma * (1.0 + lss_dishonesty_factor)
        return model(x, sigma, **extra_args)
```

For those who don't speak Python, that translates as: "if sigma is between the start and end values, 
multiply it by a number close to one before passing it on to the model".

In the description [Jonseed](https://github.com/Jonseed) writes *negative values adjust the sigmas down, increasing detail*. 
This works because a (small) negative value of `lss_dishonesty_factor` means that sigma is multiplied by a value slightly less than one,
reducing it, which tells the model there is less noise than there actually is. The model has to interpret the noisy image in a way that
is consistent with there being less noise, and one way of doing that is by adding detail to the image - fine details are
more similar to noise than large features, so assuming more detail in the image can reduce the noise that needs to be removed.


# To be added 

- techniques that add noise back in
- different noise distributions