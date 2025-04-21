# Code Examples for Transformer Study

This folder is for storing code examples from other transformer implementations to compare with TransformEw2. Below are some recommended examples to include:

## Recommended Examples

1. **The Annotated Transformer** (Harvard NLP)
   - A line-by-line implementation with explanations
   - URL: http://nlp.seas.harvard.edu/2018/04/03/attention.html
   - GitHub: https://github.com/harvardnlp/annotated-transformer

2. **Hugging Face Transformers**
   - A popular implementation with extensive documentation
   - URL: https://huggingface.co/docs/transformers/index
   - GitHub: https://github.com/huggingface/transformers

3. **PyTorch's nn.Transformer**
   - The official PyTorch implementation
   - URL: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
   - GitHub: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py

4. **Tensor2Tensor**
   - Google's original transformer implementation
   - GitHub: https://github.com/tensorflow/tensor2tensor

5. **Fairseq**
   - Facebook's sequence-to-sequence toolkit
   - GitHub: https://github.com/facebookresearch/fairseq

## How to Use These Examples

1. Download or clone the repositories
2. Extract the relevant transformer implementation files
3. Place them in this folder, organized by source
4. Compare the implementations with TransformEw2
5. Note the differences in approach, style, and optimizations

## Specific Files to Look For

### The Annotated Transformer
- `The_Annotated_Transformer.ipynb`: The main notebook with the implementation

### Hugging Face Transformers
- `modeling_bert.py`: BERT implementation
- `modeling_t5.py`: T5 implementation
- `modeling_gpt2.py`: GPT-2 implementation

### PyTorch's nn.Transformer
- `transformer.py`: The transformer implementation
- `multihead_attention.py`: Multi-head attention implementation

### Tensor2Tensor
- `transformer.py`: The original transformer implementation

### Fairseq
- `transformer.py`: Fairseq's transformer implementation
- `multihead_attention.py`: Multi-head attention implementation

By comparing these different implementations, you'll gain a deeper understanding of the transformer architecture and the various ways it can be implemented.
