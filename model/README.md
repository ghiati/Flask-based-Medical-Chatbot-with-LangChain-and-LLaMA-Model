## Download the Llama 2 Model

To use this project, you'll need the Llama 2 model. Follow these steps to download it:

1. Download the model file from [this link](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/ma). The file you need is named `llama-2-7b-chat.ggmlv3.q4_0.bin`.
2. Place the downloaded file into the `models/` directory.
3. Ensure the model file is named `llama-2-7b-chat.ggmlv3.q4_0.bin`.

Example command to download and move the file:

```bash
mkdir -p models
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main/llama-2-7b-chat.ggmlv3.q4_0.bin -O models/llama-2-7b-chat.ggmlv3.q4_0.bin

