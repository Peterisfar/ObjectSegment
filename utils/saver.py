import os
import torch


class Saver(object):

    def __init__(self):
        self.base_dir = os.path.join("result")
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def save_checkpoint(self, state, is_best):
        """Saves checkpoint to disk"""
        best_weight = os.path.join(self.base_dir, "weights", "best.pt")
        last_weight = os.path.join(self.base_dir, "weights", "last.pt")

        torch.save(state, last_weight)

        if is_best:
            torch.save(state['model'], best_weight)