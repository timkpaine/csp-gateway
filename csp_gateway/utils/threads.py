from threading import Thread

__all__ = ("get_thread",)


def get_thread(*args, **kwargs) -> Thread:
    # Returns a thread that is sure to be daemonized
    thread = Thread(*args, **kwargs)
    thread.daemon = True
    return thread
