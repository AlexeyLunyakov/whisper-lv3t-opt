# whisper-lv3t-opt

<a name="readme-top"></a>  
<div align="center">
  <p align="center">
    <h1 align="center">WLV3T-OPT</h1>
  </p>

  <p align="center">
    <p><strong>Testing basic optimization of the Whisper lv3t model.</strong></p>
    <br /><br />
  </p>
</div>

<center>

**Contents** |
:---:|
[Abstract](#title1) |
[Model](#title2) |
[Description](#title3) |
[Testing and Deployment](#title4) |
[Updates](#title5) |

</center>

## <h3 align="start"><a id="title1">Abstract</a></h3> 
So, I have a task: to analyze and test the ASR model, its optimization (and conversion to ONNX format) of the Whisper large v3 turbo model (you can find this model on HuggingFace: https://huggingface.co/openai/whisper-large-v3-turbo).


<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>

## <h3 align="start"><a id="title2">Model</a></h3>

Whisper large-v3-turbo (in lv3t or turbo laptops) is a multilingual SOTA model for automatic speech recognition and translation. Trained on 5 million hours of labeled data, the model demonstrates excellent generalization ability on data from various domains, even in zero-shot execution.

Large-v3-turbo is a fine-tune version of large-v3, in other words, it is the same model, except that the number of decoding layers has been reduced from 32 to 4, itself faster than large-v3 and distil-large-v3.

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>

## <h3 align="start"><a id="title3">Description</a></h3>

<h4 align="start"><a>WER</a></h4>

Testing of the model optimization will be carried out using the LibriSpeech ASR corpus dataset (test-clean part) and calculation of the WER metric.
* [LibriSpeech ASR corpus (OpenSLR)](https://www.openslr.org/12 )

* [JIWER](https://jitsi.github.io/jiwer/usage/ )


<h4 align="start"><a>Optimization</a></h4>

* [FP32\FP16 Conversion](https://huggingface.co/openai/whisper-large-v3-turbo)

    - Converting model weights to lower precision (FP32 to FP16) reduces VRAM usage and speeds up computations on compatible GPUs. In FP16, with lower VRAM usage and a similar WER value to FP32, we were able to reduce model validation time by 4 minutes.

* [Model weights quantization](https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c/)

    - Quantization of model parameters from FP16 to Int8 without converting the model to ONNX format also gave a small increase in speed and WER, 13 seconds and one thousandth less, respectively.

* [ONNX](https://huggingface.co/docs/transformers/v4.29.1/en/serialization), [Optimum](https://huggingface.co/docs/optimum/index), [Sherpa](https://github.com/k2-fsa/sherpa-onnx)

    - Several configurations of model translation to ONNX format were tested: Optimum + onnxoptimizer, Optimum + quantization, Sherpa ONNX.

    - Optimum is convenient both in use and in export configuration, also for large-v3-turbo there are ready-made conversions from onnx-community.

    - SherpaONNX is a bit outdated, difficult to integrate due to the need to rewrite the launch and conversion pipeline, the existing configurations only work on CPU.

* [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
    
    - Faster-whisper is a reimplementation of the Whisper model using CTranslate2 (fast inference engine for transformer models).
    
    - Several model configurations were tested, no obvious speed increase was found, also with less energy loading, the WER indicator became worse (for batched inference it is less critical). Pipeline is easy to use, although it is different from HF

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>


## <h3 align="start"><a id="title4">Testing</a></h3> 

  <br />

* wer.ipynb - final notebook with metrics and results;
* onnx.ipynb - conversion to ONNX format;
* quantization.ipynb - model quantization;

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>

## <h3 align="start"><a id="title5">Results</a></h3> 

Model conf | WER | WER-time | Latency | VRAM | CPU |
--- |:---:|:---:|:---:|:---:|:---:|
Baseline FP32 (base model, torch.float32)|0.031|12 min 51 sec|x1.00|3.258 GB| 69% |
Baseline FP16 (base model, torch.float16)|0.031|9 min 12 sec|x0.72|1.597 GB| 71% |
**Quantized Int8** (torch, loaded weights dequantize to fp16)|0.030|8 min 59 sec|x0.70|1.597 GB| 79% |
ONNX FP16 (ORTModel (inference only))|0.031|25 min 5 sec|x1.95|73.36 MB| 36% |
ONNX QUInt8 (ORTModel + ORTQuantizer + AutoQuantization)|0.032|131 min 04 sec|x10.2|71.32 MB| 95% |
ONNX Int8 (Sherpa ONNX)|0.033|149 min 03 sec|x11.6|0.00 MB| 42% |
ONNX FP16 (Sherpa ONNX)|0.033|203 min 01 sec|x15.8|0.00 MB| 44% |
Faster-Whisper FP16 (Base Inference)|0.145|14 min 26 sec|x1.12|~ 1.900 GB| 7% |
Faster-Whisper Int8 (Base Inference)|0.114|13 min 57 sec|x1.09|~ 1.200 GB| 7% |
Faster-Whisper FP16 (Batched Inference)|0.039|12 min 50 sec|x1.00|~ 1.800 GB| 8% |
Faster-Whisper Int8 (Batched Inference)|0.039|12 min 19 sec|x0.96|~ 1.000 GB| 7% |

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>


<a name="readme-top"></a>
