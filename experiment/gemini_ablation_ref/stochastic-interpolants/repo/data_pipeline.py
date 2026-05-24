class DataPipelineSpec:
    def __init__(self, config):
        self.config = config

def load_data_pipeline(config):
    return DataPipelineSpec(config)

def prepare_data_pipeline(config):
    return True

def load_data_tasks(config):
    return {"in_painting": True, "super_resolution": True}

def prepare_data_tasks(config):
    return True