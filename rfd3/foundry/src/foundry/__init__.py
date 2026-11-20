import logging
import os

import torch
from beartype.claw import beartype_this_package
from environs import Env
from jaxtyping import install_import_hook

_env = Env()
_env.read_env()
should_typecheck = _env.bool("TYPE_CHECK", default=False)
should_debug = _env.bool("DEBUG", default=False)
should_check_nans = _env.bool("NAN_CHECK", default=True)

logger = logging.getLogger("foundry")
logger.setLevel(logging.DEBUG if should_debug else logging.INFO)
logger.debug("Debug mode: %s", should_debug)
logger.debug("Type checking mode: %s", should_typecheck)
logger.debug("NAN checking mode: %s", should_check_nans)

if should_typecheck:
    beartype_this_package()
    install_import_hook("foundry", "beartype.beartype")

SHOULD_USE_CUEQUIVARIANCE = False

try:
    if torch.cuda.is_available():
        if _env.bool("DISABLE_CUEQUIVARIANCE", default=False):
            logger.info("cuEquivariance usage disabled via DISABLE_CUEQUIVARIANCE")
        else:
            import cuequivariance_torch as cuet

            SHOULD_USE_CUEQUIVARIANCE = True
            os.environ["CUEQ_DISABLE_AOT_TUNING"] = _env.str(
                "CUEQ_DISABLE_AOT_TUNING", default="1"
            )
            os.environ["CUEQ_DEFAULT_CONFIG"] = _env.str(
                "CUEQ_DEFAULT_CONFIG", default="1"
            )
            logger.info("cuEquivariance is available and will be used.")

except ImportError:
    logger.debug("cuEquivariance unavailable: import failed")


DISABLE_CHECKPOINTING = False

__all__ = ["SHOULD_USE_CUEQUIVARIANCE", "DISABLE_CHECKPOINTING"]
