## Sample Code for our proposed SGA
SGA adapts a pretrained multilingual understanding model, XLMR, to a NAR generator in a parameter-efficient way.

### Implementation & Requirements
Our implementation is based on a public available codebase, https://github.com/THUNLP-MT/PLM4MT , which adapts a pretrained autoregressive model(mGPT) to translation tasks via multi-stage prompting (MSP). Please refer to https://github.com/THUNLP-MT/PLM4MT for detailed usage of codebase

In the sample code, we provide the model script, the data processing script, and the training script that differs from MSP, and we provide a simple train script for reference.
