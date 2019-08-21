from easyagents import core


class Count(core.TrainCallback):

    def __init__(self):
        self.train_begin_count = 0
        self.train_end_count = 0
        self.iteration_begin_count = 0
        self.iteration_end_count = 0

    def __str__(self):
        return f'train={self.train_begin_count}:{self.train_end_count} ' + \
               f'iteration={self.iteration_begin_count}:{self.iteration_end_count}'

    def on_train_begin(self, train_context: core.TrainContext):
        """Called once at the entry of an agent.train() call. """
        self.train_begin_count += 1

    def on_train_end(self, train_context: core.TrainContext):
        """Called once before exiting an agent.train() call"""
        self.train_end_count += 1

    def on_iteration_begin(self, train_context: core.TrainContext):
        """Called once at the start of a new iteration. """
        self.iteration_begin_count += 1

    def on_iteration_end(self, train_context: core.TrainContext):
        """Called once after the current iteration is completed"""
        self.iteration_end_count += 1
