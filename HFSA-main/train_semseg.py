"""Backward-compatible entry for the renamed referring-segmentation task.

New commands should use ``train_refseg.py`` and ``test_refseg.py``. Public
helpers remain re-exported so existing experiments and tests keep working.
"""

from tasks.refseg.engine import *  # noqa: F401,F403
from tasks.refseg.engine import main


if __name__ == "__main__":
    main()
