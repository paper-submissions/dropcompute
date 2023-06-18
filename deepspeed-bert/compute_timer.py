import torch
import utils
import time
from collections import deque


class ComputeTimeout(Exception):
    pass


class DeviceTimer(object):
    """This module is in charge if measuring time on the host while being aware
     of the device state. self.events is a cyclic queue that keeps track of the
     last N events on the compute stream. each time _sync is called, an event
     is added to the compute stream, and pushed last to the cyclic queue.
     Then, the host is synchronized on the LAST event that exists in the cyclic
     queue to make sure it was executed on the device. In fact, the _sync
     method is a delayed sync, with delay=N. N can be controlled by the
     parameter event_sync_delay.
    """
    def _get_hpu_module(self):
        if self.habana_module is None:
            import habana_frameworks.torch as ht
            self.habana_module = ht

    def __init__(self, use_hpu=False, event_sync_delay=0):
        self.start_time = None
        self.events = deque(maxlen=event_sync_delay+1)  # sync on the previous event according to parameter
        self.habana_module = None
        self.use_hpu = use_hpu
        self.enable_drop_compute = False
        self.dropped = False
        self.drop_threshold = 0
        if (use_hpu):
            self._get_hpu_module()

    def _sync(self):
        if self.use_hpu:
            sync_event = self.habana_module.hpu.Event(enable_timing=True)
            sync_event.record()
            self.habana_module.hpu.current_stream().wait_event(sync_event)
        else:
            sync_event = torch.cuda.Event(enable_timing=True)
            sync_event.record()
            torch.cuda.current_stream().wait_event(sync_event)

        self.events.append(sync_event)      # push event onto cyclic queue
        wait_event = self.events[0]         # get the last (delayed) event on the queue
        wait_event.synchronize()            # synchronize with the last event

    def start(self):
        self._sync()
        self.start_time = time.time()

    def set_delay(self, delay):
        self.events = deque(maxlen=delay+1)

    def elapsed(self):
        if self.start_time is None:
            return 0
        self._sync()
        return time.time() - self.start_time

    def elapsed_no_sync(self):
        return time.time() - self.start_time

    def check_drop_compute_throw(self, name=""):
        self._sync()
        if not self.is_started():
            return False

        current_time = self.elapsed()
        if self.enable_drop_compute and (current_time > self.drop_threshold):
            self.dropped = True
            raise ComputeTimeout('')

    def reset(self):
        self.dropped = False
        self.start_time = None

    def is_started(self):
        return self.start_time is not None


class WallClockTimer(object):
    """This module is in charge if measuring time on the host while being aware
     of the device state. self.events is a cyclic queue that keeps track of the
     last N events on the compute stream. each time _sync is called, an event
     is added to the compute stream, and pushed last to the cyclic queue.
     Then, the host is synchronized on the LAST event that exists in the cyclic
     queue to make sure it was executed on the device. In fact, the _sync
     method is a delayed sync, with delay=N. N can be controlled by the
     parameter event_sync_delay.
    """
    def _get_hpu_module(self):
        if self.habana_module is None:
            import habana_frameworks.torch as ht
            self.habana_module = ht

    def __init__(self, use_hpu=False):
        self.start_time = None
        self.stop_time = None
        self.habana_module = None
        self.use_hpu = use_hpu
        if (use_hpu):
            self._get_hpu_module()

    def _sync(self):
        if self.use_hpu:
            sync_event = self.habana_module.hpu.Event(enable_timing=True)
            sync_event.record()
            self.habana_module.hpu.current_stream().wait_event(sync_event)
        else:
            sync_event = torch.cuda.Event(enable_timing=True)
            sync_event.record()
            torch.cuda.current_stream().wait_event(sync_event)

        sync_event.synchronize()

    def start(self):
        self._sync()
        self.start_time = time.time()

    def stop(self):
        self._sync()
        self.stop_time = time.time()

    def elapsed(self, reset=False):
        if self.start_time is None:
            return 0
        if self.stop_time is None:
            self.stop()
        result = self.stop_time - self.start_time
        if reset:
            self.reset()
        return result * 1000  # seconds to ms

    def reset(self):
        self.start_time = None
        self.stop_time = None

    def is_started(self):
        return self.start_time is not None