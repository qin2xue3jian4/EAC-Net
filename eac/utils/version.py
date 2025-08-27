try:
    from importlib.metadata import metadata

    meta = metadata("eacnet")
except ImportError:
    meta = {
        'Summary': 'An open-source code for predicting atomic contributions ' \
            'charge density using Equivariant Message Passing Networks.',
        'Version': '0.1.0',
    }

class SoftwareInfo:
    fullname = 'Real-space Equivariant Atomic Contributions Network'
    name = 'EAC-Net'
    description = meta['Summary']
    homepage = 'https://github.com/qin2xue3jian4/EAC-Net'
    __version__ = meta['Version']
    ascii_img = """
  ______          _____      _   _      _   
 |  ____|   /\   / ____|    | \ | |    | |  
 | |__     /  \ | |   ______|  \| | ___| |_ 
 |  __|   / /\ \| |  |______| . ` |/ _ \ __|
 | |____ / ____ \ |____     | |\  |  __/ |_ 
 |______/_/    \_\_____|    |_| \_|\___|\__|
                                            
"""
