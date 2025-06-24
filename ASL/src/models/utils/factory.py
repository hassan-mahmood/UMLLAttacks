# import logging

# logger = logging.getLogger(__name__)

# # from ..tresnet import TResnetM, TResnetL, TResnetXL
# from ..tresnet import TResnetM, TResnetL



# def create_model(args):
#     """Create a model
#     """
#     model_params = {'args': args, 'num_classes': args.num_classes}
#     args = model_params['args']
#     args.model_name = args.model_name.lower()

#     if args.model_name=='tresnet_m':
#         model = TResnetM(model_params)
#     elif args.model_name=='tresnet_l':
#         model = TResnetL(model_params)
#     elif args.model_name=='tresnet_xl':
#         model = TResnetXL(model_params)
#     else:
#         print("model: {} not found !!".format(args.model_name))
#         exit(-1)

#     return model


import logging
import os
from urllib import request

import torch

logger = logging.getLogger(__name__)

from ..tresnet.tresnet import TResnetM, TResnetL
# import sys 
# sys.path.append('ML_Decoder/src_files/')
from ...mldecoder.ml_decoder import add_ml_decoder_head


def create_model(args):
    """Create a model
    """
    load_head=args.load_head
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    
    if args.model_name == 'tresnetm_mldecoder':
        model = TResnetM(model_params)
        
    elif args.model_name == 'mldecoder' or args.model_name=='asl':
        model = TResnetL(model_params)
    elif args.model_name == 'tresnet_xl':
        model = TResnetXL(model_params)
    else:
        0/0
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    ####################################################################################
    if args.use_ml_decoder:
        model = add_ml_decoder_head(model,num_classes=args.num_classes,num_of_groups=args.num_of_groups,
                                    decoder_embedding=args.decoder_embedding, zsl=args.zsl)

    ####################################################################################
    # loading pretrain model

    # model_path = args.model_path
    # print('Loading model')
    # if args.use_ml_decoder:
    #     if os.path.exists(model_path):  # make sure to load pretrained model
            
    #         state = torch.load(model_path, map_location='cpu')

    #         if not load_head:
    #             if 'model_state' in state.keys():
    #                 key = 'model_state'
    #             elif 'model' in state:
    #                 key = 'model'
    #             else:
    #                 key = 'state_dict'
    #             filtered_dict = {k: v for k, v in state[key].items() if
    #                              (k in model.state_dict() and 'head.fc' not in k)}
    #             model.load_state_dict(filtered_dict, strict=False)
    #         else:
    #             0/0
    #             # pass
    #             model.load_state_dict(state['model'], strict=True)

    #         print('Loaded model')

    return model
