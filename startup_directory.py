import os

class make_startup_directory():
  def __init__(self):
    self.output_path = '../'
    #if not os.path.exists(self.output_path):
    #  os.makedirs(self.output_path)
    subpaths = ['figs', 'dicts', 'models', 'params', 'preprocessed', 'submissions']
    self.output_dict = dict.fromkeys(subpaths)
    for pth in subpaths:
      output_subpath = os.path.join(self.output_path, pth)
      self.output_dict[pth] = output_subpath
      if not os.path.exists(output_subpath):
        os.makedirs(output_subpath)
