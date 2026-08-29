# Server layout - 8x A100-SXM4-80GB @ 64.181.219.88

  ~/areal-selfevo/   OUR fork of AReaL (Apache-2.0). All code we write lives here.
  ~/baselines/       Third-party, cloned to RUN IN PLACE. Never copied into our repo.
  ~/runs/            Experiment outputs, one subdir per run.
  ~/scratch/         Throwaway.
  ~/venv312b/        py3.12.14, torch 2.9.1+cu128, sglang 0.5.10.post1
                     export PATH=$HOME/.local/bin:$PATH; source ~/venv312b/bin/activate

## Pinned baselines

  Absolute-Zero-Reasoner     484afa4    LICENSE
  BigBang-v1                 5128884    LICENSE
  DAMO-ConvAI                483554e    LICENSE
  MEDS                       4f46841    LICENSE
  R-Zero                     5699329    NO-LICENSE
  RAGEN                      d97bb32    LICENSE
  Search-R1                  598e61b    LICENSE
  Spider2                    cafb8673   LICENSE
  autoresearch               228791f    NO-LICENSE
  nanochat                   92d63d4    LICENSE

## Rule

NO-LICENSE repos (autoresearch, R-Zero) are cloned to RUN and CITE only. Default
copyright means no redistribution right regardless of public visibility. Their code
is never copied into ~/areal-selfevo or any repo we publish.
