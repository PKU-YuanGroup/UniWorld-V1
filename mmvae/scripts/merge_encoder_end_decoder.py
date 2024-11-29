import torch

encoder = "/storage/lcm/WF-VAE_paper/results/wfvae-16dim-large/merge.ckpt"
decoder = "/storage/lcm/WF-VAE_paper/results/FORMAL_16DIM_L_ont_resume_dic_onlydecoder-128_encode_128_decode-lr1.00e-05-bs1-rs256-sr5-fr37/checkpoint-1005000.ckpt"
output = "/storage/lcm/WF-VAE_paper/results/wfvae-16dim-large/merge2.ckpt"

encoder_ckpt = torch.load(encoder, map_location="cpu")
decoder_ckpt = torch.load(decoder, map_location="cpu")

new_ckpt = encoder_ckpt
# print(encoder_ckpt.keys())


for key, param in decoder_ckpt['ema_state_dict'].items():
    new_ckpt['ema_state_dict'][key] = param

torch.save(new_ckpt, output)