
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
NF4=torch.tensor([-1.0,-0.6961928,-0.5250731,-0.3949175,-0.2844414,-0.1847734,-0.0910500,0.0,
 0.0795803,0.1609302,0.2461123,0.3379152,0.4407098,0.5626170,0.7229568,1.0])
def gpad(x,g):
    n=x.shape[2]; pad=(g-n%g)%g
    if pad: x=torch.cat([x,x[:,:,-1:,:].expand(-1,-1,pad,-1)],2)
    return x,n
def nf4g(x,g=32):
    B,H,n,D=x.shape; xp,n0=gpad(x,g); Tg=xp.shape[2]//g; xg=xp.reshape(B,H,Tg,g,D)
    amax=xg.abs().amax(3,keepdim=True).clamp_min(1e-8); xn=xg/amax
    lvl=NF4.to(x.device).view(1,1,1,1,1,-1); idx=(xn.unsqueeze(-1)-lvl).abs().argmin(-1)
    return (NF4.to(x.device)[idx]*amax).reshape(B,H,Tg*g,D)[:,:,:n0,:]
def unig(x,bits=4,g=32):
    B,H,n,D=x.shape; xp,n0=gpad(x,g); Tg=xp.shape[2]//g; xg=xp.reshape(B,H,Tg,g,D)
    mn=xg.amin(3,keepdim=True); mx=xg.amax(3,keepdim=True); qm=2**bits-1
    sc=(mx-mn).clamp_min(1e-8)/qm
    return (((xg-mn)/sc).round().clamp(0,qm)*sc+mn).reshape(B,H,Tg*g,D)[:,:,:n0,:]
def relerr(x,xq): return ((x-xq).float().norm()/x.float().norm()).item()
def analyze(mid):
    m=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float16,attn_implementation="eager",trust_remote_code=True).cuda().eval()
    try: tok=AutoTokenizer.from_pretrained(mid,use_fast=True)
    except: tok=AutoTokenizer.from_pretrained(mid,use_fast=False)
    cap={}
    def mk(li):
        def h(mod,inp,out):
            o=out.detach()
            nkv=mod.weight.shape[0]//(m.config.hidden_size//m.config.num_attention_heads)
            D=m.config.hidden_size//m.config.num_attention_heads
            cap[li]=o.view(1,o.shape[1],nkv,D).permute(0,2,1,3)  # (1,nkv,T,D)
        return h
    hs=[m.model.layers[li].self_attn.k_proj.register_forward_hook(mk(li)) for li in [0,10,20]]
    ids=tok("The quick brown fox. "*120, return_tensors="pt").input_ids.cuda()[:,:1024]
    with torch.no_grad(): m(ids)
    for h in hs: h.remove()
    print(f"== {mid} ==")
    for li in [0,10,20]:
        K=cap[li].float()  # (1,nkv,T,D) pre-RoPE keys
        e_nf4=relerr(K,nf4g(K)); e_uni=relerr(K,unig(K,4))
        # per-32-group offset vs signal: within-group (max-min) vs absmax, per channel
        B,H,n,D=K.shape; g=32; Kp,_=gpad(K,g); Tg=Kp.shape[2]//g; Kg=Kp.reshape(B,H,Tg,g,D)
        rng=(Kg.amax(3)-Kg.amin(3)); amx=Kg.abs().amax(3).clamp_min(1e-8)
        sig2off=(rng/amx)  # ~1 => signal fills range (NF4 ok); ~0 => offset dominates (NF4 bad)
        print(f" L{li}: NF4 relerr={e_nf4:.4f}  uniform4 relerr={e_uni:.4f}  ratio={e_nf4/e_uni:.2f}x  | per-grp range/absmax median={sig2off.median():.3f}")
    del m; torch.cuda.empty_cache()
analyze("Qwen/Qwen2.5-7B-Instruct")
analyze("NousResearch/Llama-2-7b-chat-hf")
