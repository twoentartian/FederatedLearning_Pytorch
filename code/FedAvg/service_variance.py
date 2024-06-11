class Service(object):
    pass


def calculate_weights_variance(model):
    import torch
    layer_variances = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weights = module.weight.data
            variance = torch.var(weights).item()
            layer_variances[name] = variance
    
    return layer_variances


class ServiceVarianceRecorder(Service):
    def __init__(self, save_path) -> None:
        import os
        super().__init__()
        self.save_path = save_path
        self.save_file = open(os.path.join(save_path, "variance.csv"), "w+")
        self.header_order = None
        

    def write_header(self, model):
        all_names = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                all_names.append(name)
        header = ",".join(all_names)
        self.save_file.write(f"{header}\n")
        self.save_file.flush()
        self.header_order = all_names


    def write_row(self, model):
        import torch
        layer_variances = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weights = module.weight.data
                variance = torch.var(weights).item()
                layer_variances[name] = variance
        
        row_value = []
        for name in self.header_order:
            if name in layer_variances:
                row_value.append(f'{layer_variances[name]:e}')
        row = ",".join(row_value)
        self.save_file.write(f"{row}\n")
        self.save_file.flush()


    def __del__(self):
        self.save_file.flush()
        self.save_file.close()