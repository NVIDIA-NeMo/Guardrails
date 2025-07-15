# AlignScore Deployment

```{note}
The recommended way to use AlignScore with NeMo Guardrails is using the provided [Dockerfile](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/nemoguardrails/library/factchecking/align_score/Dockerfile). For more details, check out how to [build and use the image](using-docker.md).
```

In order to deploy an AlignScore server, follow these steps:

```{important}
Installing AlignScore is not supported on Python 3.11.
```

1. Install the `alignscore` package from the GitHub repository:

```bash
git clone https://github.com/yuh-zha/AlignScore.git
cd AlignScore
pip install .
```

2. Install Pytorch version `2.0.1`.

```bash
pip install torch==2.0.1
```

3. Download the Spacy `en_core_web_sm` model:

```bash
python -m spacy download en_core_web_sm
```

4. Download the one or both of the AlignScore checkpoints:
```
curl -OL https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt
curl -OL https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt
```

5. Set the `ALIGN_SCORE_PATH` environment variable to point to the path where the checkpoints have been downloaded.

6. Set the `ALIGN_SCORE_DEVICE` environment variable to `"cpu"` to run the AlignScore model on CPU, or to the corresponding GPU device, e.g. `"cuda:0"`.
```bash
export ALIGN_SCORE_PATH=<path/to/folder_containing_ckpt>
export ALIGN_SCORE_DEVICE="cuda:0"
```

7. Start the AlignScore server.

```bash
python -m nemoguardrails.library.factchecking.align_score.server --port 5000 --models=base
```

By default, the AlignScore server listens on port `5000`. You can change the port using the `--port` option. Also, by default, the AlignScore server loads only the base model. You can load only the large model using `--models=large` or both using `--models=base --models=large`.
