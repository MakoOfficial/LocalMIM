import models_mim
import torch
# model = models_mim.__dict__['MIM_vit_base_patch16'](hog_nbins=9, hog_bias=False)
#     # 加入预训练
# checkpoint = torch.load('./vit_base_localmim_hog_1600ep_pretrain.pth', map_location='cpu')
# print("Load pre-trained checkpoint")
# checkpoint_model = checkpoint['model']
# state_dict = model.state_dict()
# for k in ['head.weight', 'head.bias']:
#     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#         print(f"Removing key {k} from pretrained checkpoint")
#         del checkpoint_model[k]
#
# # load pre-trained model
# msg = model.load_state_dict(checkpoint_model, strict=True)
# print(msg)

# finetune
import models_vit
from util.pos_embed import interpolate_pos_embed
from classifier import Classifier
model = models_vit.__dict__["vit_base_patch16"](num_classes=230, drop_path_rate=0.1, global_pool=True)
checkpoint = torch.load("checkpoint-100.pth", map_location='cpu')
print("Load pre-trained checkpoint from: %s" % "ViT/checkpoint-100.pth")
checkpoint_model = checkpoint['model']
# print(checkpoint_model.keys())
state_dict = model.state_dict()
# print(state_dict.keys())

state_dict.update({k:v for k,v in checkpoint_model.items() if k in state_dict.keys()})

# interpolate position embedding
interpolate_pos_embed(model, state_dict)

# load pre-trained model
msg = model.load_state_dict(state_dict, strict=False)
print(msg)

# assert set(msg.missing_keys) == {'fc_norm.weight', 'fc_norm.bias'}

# manually initialize fc layer
# trunc_normal_(model.head.weight, std=2e-5)
model = Classifier(model, 768)
print(model.state_dict().keys())
print("load success")

data = torch.ones((2, 3, 224, 224))
gender = torch.ones((2, 1))

output = model(data, gender)
print(output.shape)
