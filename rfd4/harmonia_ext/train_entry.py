"""``rfd4-train`` with the HARMONIA ADJ class registered (same Hydra CLI; configs in harmonia_ext/configs on the searchpath).

  python -m harmonia_ext.train_entry experiment=harmonia_r1 hf_ckpt_path=/w/harmonia_rfd4/base out_dir=<node-local> ...

At import (every process, incl. spawned DataLoader workers): the model class + collate spec (``adj_model.register``). Main process only:
LoRA unfreeze of the new parameters (``adj_embed``, RFD4's SS embedding) after PEFT freezes the model, two-group Adam (new parameters at
``opt.new_module_lr_mult`` x lr, kept off Muon), and a guard that refuses to train without the expanded base bundle (= from scratch).
Generic optimiser glue is reused from ``triadj_ext.s2.optim_glue`` (no RING code is involved).
"""
import os
import sys

from harmonia_ext.adj_model import register

register()

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
NEW_PATTERNS = [r"^nn\.adj_embed\.", r"ss_embedding"]


def _guard(orig):
    def get_model_n_ckpt_resume(cfg_exp, ckpt_path_store):
        from rfproteina.utils.weights import fetch_last_ckpt

        if not (cfg_exp.get("hf_ckpt_path") or cfg_exp.get("resume_ckpt_path")) and fetch_last_ckpt(ckpt_path_store) is None:
            raise ValueError("no hf_ckpt_path (expanded base bundle) and nothing to resume: this would train from scratch")
        return orig(cfg_exp, ckpt_path_store)

    return get_model_n_ckpt_resume


def main() -> None:
    if not any(a.startswith(("hydra.searchpath=", "+hydra.searchpath=")) for a in sys.argv):
        sys.argv.insert(1, f"hydra.searchpath=[file://{CONFIG_DIR}]")
    import rfproteina.train as T

    from triadj_ext.s2 import optim_glue

    T.get_model_n_ckpt_resume = _guard(T.get_model_n_ckpt_resume)
    optim_glue.install_lora_unfreeze(lambda: NEW_PATTERNS)
    optim_glue.install_two_group_adam(NEW_PATTERNS)
    T.main()


if __name__ == "__main__":
    main()
