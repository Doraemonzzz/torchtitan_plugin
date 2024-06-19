# torchtitan plugin
Some plugins for using [torchtitan](https://github.com/pytorch/torchtitan).

## Todo
- [x] Load local dataset.
- [ ] Support wandb.
- [ ] Add gnorm stat.
  - [ ] https://github.com/pytorch/torchtitan/issues/119
- [ ] Support pp.
- [ ] Test dp.
- [ ] Test fsdp.
- [ ] Test tp.
- [x] Support gpt2 tokenzer, hf tokenzer.

## benchmark
The benchmark results for speed, all tested with two A800 80g gpus.

| model      | tgs-torchtitan | memory-torchtitan | tgs-metaseq | memory-metaseq | bs | seqlen | config   |
|------------|----------------|-------------------|-------------|----------------|----|--------|----------|
| llama-465m | 40k            | 70                | 30k         | 62.9           | 36 | 2k     | fsdp+act |
| llama-7b   | 3.6k           | 70                | 2.5k        | 75             | 8  | 2k     | fsdp+act |
