"""Publish complete immutable trees atomically, including added/removed paths."""
import os
from pathlib import Path
import shutil
from storage_budget import require_space


class GenerationStore:
    def __init__(self, directory, sources):
        self.directory = Path(directory)
        if self.directory.exists():
            raise FileExistsError('Retain previous generations; choose a fresh directory')
        require_space(512 * 1024**2)
        self.directory.mkdir()
        self.trees = []
        for index, source in enumerate(sources):
            target = self.directory/f'build-{index}'
            shutil.copytree(source, target, copy_function=os.link)
            self.trees.append(target)
        self.current = self.directory/'current'
        self.publish(0)

    def publish(self, index):
        target = self.trees[index]
        pending = self.directory/'next'
        if pending.exists() or pending.is_symlink():
            raise FileExistsError('Unfinished generation publication remains')
        pending.symlink_to(target.name, target_is_directory=True)
        os.replace(pending, self.current)

    def retain_origin_marker(self, identity):
        for tree in self.trees:
            path = tree/identity['marker']
            if not path.exists():
                path.write_text(identity['identity'])
