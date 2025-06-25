BLOCK_REGISTRY = {}

def register_block(name):
    def decorator(cls):
        BLOCK_REGISTRY[name] = cls
        return cls
    return decorator
