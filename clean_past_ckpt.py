# %%
import os
from data_helper import find_lastest_checkpoint
root_dir = os.path.abspath(os.path.dirname(__file__))
for model_dir in os.listdir(os.path.join(root_dir, 'save_dir')):
    for version_dir in os.listdir(os.path.join(root_dir, 'save_dir', model_dir)):
        cdir = os.path.join(root_dir, 'save_dir', model_dir, version_dir, 'checkpoints')
        keeped = find_lastest_checkpoint(cdir)
        for file_name in os.listdir(cdir):
            full_path = os.path.join(cdir, file_name)
            if full_path != keeped:
                os.remove(full_path)

# %%
