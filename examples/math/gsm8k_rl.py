import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_tokens=config.gconfig.max_tokens,
        max_completion_tokens=config.gconfig.max_new_tokens,
        # Force at least one non-EOS token. As entropy falls the policy begins sampling
        # EOS first, yielding an all-EOS/PAD completion. sglang then returns 500
        # "All output_tokens are EOS or PAD tokens; cannot strip stop tokens without
        # removing entire output"; the workflow catches it and AReaL scores the failed
        # trajectory as reward 0.0. That feeds a false zero back into training and pushes
        # entropy lower still -- the feedback loop that produced 3,468,076 such errors in
        # step0, and 32 within one step of step0c once entropy reached 0.03.
        #
        # The field name matters. sglang's OpenAI adapter declares `min_tokens` on
        # ChatCompletionRequest and maps it in to_sampling_params() as
        #     "min_new_tokens": self.min_tokens          (protocol.py:771)
        # so sending "min_new_tokens" here would be silently ignored. AReaL's own
        # gconfig.min_new_tokens is a dead field (declared in cli_args.py, read nowhere),
        # which is why this is passed through extra_body rather than the config.
        extra_body={"min_tokens": 1},
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.openai.math_agent.MathAgent",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.openai.math_agent.MathAgent",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
