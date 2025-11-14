from transformers import TrainerCallback
import tempfile, os
from huggingface_hub import HfApi

class PushToHubEachEpoch(TrainerCallback):
    def __init__(self, repo_id, run_prefix="", include_resume_state=True):
        self.api = HfApi()
        self.repo_id = repo_id
        self.run_prefix = run_prefix
        self.include_resume_state = include_resume_state

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        tokenizer = kwargs.get("tokenizer", None)
        epoch = int(state.epoch or 0)
        subpath = f"{self.run_prefix}checkpoints/epoch-{epoch:02d}"

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir, safe_serialization=True, max_shard_size="2GB")
            if tokenizer: tokenizer.save_pretrained(tmpdir)
            # Save resume state if you want to be able to resume:
            if self.include_resume_state:
                state.save_to_json(os.path.join(tmpdir, "trainer_state.json"))
                # If using Trainer's default, optimizer/scheduler are in output_dir/checkpoint-*;
                # consider manually saving them here if you need exact resume.

            self.api.upload_folder(
                repo_id=self.repo_id,
                folder_path=tmpdir,
                path_in_repo=subpath,
                commit_message=f"epoch {epoch} checkpoint",
                # IMPORTANT: do NOT ignore trainer_state/optimizer if you need resume
                ignore_patterns=["**/.git/*", "**/.ipynb_checkpoints/*"],
            )
        return control
