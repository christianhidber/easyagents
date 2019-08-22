from typing import Tuple

from easyagents import core as core


class Count(core.ApiCallback):

    def __init__(self):
        self.gym_init_begin_count = 0
        self.gym_init_end_count = 0
        self.gym_reset_begin_count = 0
        self.gym_reset_end_count = 0
        self.gym_step_begin_count = 0
        self.gym_step_end_count = 0
        self.backend_call_begin_count = 0
        self.backend_call_end_count = 0

    def __str__(self):
        return f'backend_call={self.backend_call_begin_count}:{self.backend_call_end_count} ' + \
               f'gym_init={self.gym_init_begin_count}:{self.gym_init_end_count} ' + \
               f'gym_reset={self.gym_reset_begin_count}:{self.gym_reset_end_count} ' + \
               f'gym_step={self.gym_step_begin_count}:{self.gym_step_end_count}'

    def on_backend_call_begin(self, call_name: str, api_context: core.ApiContext):
        self.backend_call_begin_count += 1

    def on_backend_call_end(self, call_name: str, api_context: core.ApiContext):
        self.backend_call_end_count += 1

    def on_gym_init_begin(self, api_context: core.ApiContext):
        self.gym_init_begin_count += 1

    def on_gym_init_end(self, api_context: core.ApiContext):
        self.gym_init_end_count += 1

    def on_gym_reset_begin(self, api_context: core.ApiContext, **kwargs):
        self.gym_reset_begin_count += 1

    def on_gym_reset_end(self, api_context: core.ApiContext, reset_result: Tuple, **kwargs):
        self.gym_reset_end_count += 1

    def on_gym_step_begin(self, api_context: core.ApiContext, action):
        self.gym_step_begin_count += 1

    def on_gym_step_end(self, api_context: core.ApiContext, action, step_result: Tuple):
        self.gym_step_end_count += 1
