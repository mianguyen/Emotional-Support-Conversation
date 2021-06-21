"""Contains encoders and decoders"""

from configuration import BlenderbotSmallConfig
from modeling.blenderbot_small import MyBlenderbotSmallEncoder


encoders = {
    'blenderbot_small': (BlenderbotSmallConfig, MyBlenderbotSmallEncoder),
}


from modeling.blenderbot_small import MyBlenderbotSmallDecoder

decoders = {
    'blenderbot_small': (BlenderbotSmallConfig, MyBlenderbotSmallDecoder),
}

