"""Competition convenience entry.

Default behavior runs inference/submission generation. Pass normal infer.py
arguments, for example:
    python 1.py --data-dir data --output-dir outputs/track1/submission
"""

from infer import main


if __name__ == "__main__":
    main()
