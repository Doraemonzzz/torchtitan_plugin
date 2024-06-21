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
- [ ] Integrate xmixers.
  - [x] Add llama.
- [ ]




## benchmark
The benchmark results for speed, all tested with two A800 80g gpus.

small (465m, 36, 2k, fsdp+act):

| desc       | tgs | memory |
|------------|-----|--------|
| metaseq    | 30k | 62.9   |
| torchtitan | 40k | 70     |
| xmixers    | 43k | 44.64  |


large(7b, 8, 2k, fsdp+act):

| desc       | tgs  | memory |
|------------|------|--------|
| metaseq    | 2.5k | 75     |
| torchtitan | 3.6k | 76     |
| xmixers    | 3.8k | 71~76  |
