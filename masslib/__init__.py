from . import brutto
from . import brutto_generator
from . import utils
from . import mass
from . import tmds

# import importlib
# import pkgutil


# def import_submodules(package_name):
#     """ """
#     results = {}
    
#     package = importlib.import_module(package_name)
    
#     for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
#         full_name = package.__name__ + '.' + name
#         results[full_name] = importlib.import_module(full_name)
#         #results.update(import_submodules(full_name))
        
#     return results

# import_submodules(__name__)