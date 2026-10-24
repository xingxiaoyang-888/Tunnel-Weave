import torch
from torch import nn
from torch.nn import functional as F
from .model import MainlineModel
from .contracts import require
class ImageQueries(nn.Module):
 """Image-conditioned queries for13 classes +2 auxiliary slots; no text features."""
 def __init__(self,language_dim,channels=256):
  super().__init__()
  self.queries=nn.Parameter(torch.randn(15,channels)*.02)
  self.attention=nn.MultiheadAttention(channels,4,dropout=0,batch_first=True)
  self.norm=nn.LayerNorm(channels)
  self.out=nn.Linear(channels,language_dim)
 def forward(self,image_map):
  require(image_map.ndim==4 and image_map.shape[1]==256,'SAM embedding shape')
  x=F.adaptive_avg_pool2d(image_map.float(),(16,16)).flatten(2).transpose(1,2)
  # Fixed normalized coordinates, independent of labels and image source.
  y,z=torch.meshgrid(torch.linspace(-1,1,16,device=x.device),torch.linspace(-1,1,16,device=x.device),indexing='ij')
  freq=torch.arange(1,65,device=x.device,dtype=torch.float32)[None]
  pos=torch.cat([(y.flatten()[:,None]*freq).sin(),(y.flatten()[:,None]*freq).cos(),(z.flatten()[:,None]*freq).sin(),(z.flatten()[:,None]*freq).cos()],1)[None]
  q=self.queries[None].expand(x.shape[0],-1,-1)
  a,_=self.attention(q,x+pos,x,need_weights=False)
  return self.out(self.norm(q+a))
class PaperModel(MainlineModel):
 def __init__(self,base,arm):
  super().__init__(base);self.arm=arm
  self.visual_queries=None
  if arm=='Q_no_semantic_guidance':
   with torch.random.fork_rng(devices=([next(base.parameters()).device.index] if next(base.parameters()).is_cuda else [])):
    torch.manual_seed(1701)
    self.visual_queries=ImageQueries(base.qwen.config.text_config.hidden_size).to(next(base.main_projection.parameters()).device)
 def forward(self,image,input_ids,pixel_values,image_grid_thw,attention_mask=None):
  if self.arm!='Q_no_semantic_guidance':return super().forward(image,input_ids,pixel_values,image_grid_thw,attention_mask)
  # Fixed interface is retained, but neither token IDs nor Qwen image/hidden states
  # enter this visual prediction route.
  f=self.base.sam.encode_once(image);states=self.visual_queries(f.image_embed)
  masks=[self.base.sam.decode(f,self.base.main_projection(s)[None,None])for s in states[0,:13]]
  aux=[self.base.sam.decode(f,self.base.aux_projection(s)[None,None],self.base.aux_decoder)for s in states[0,13:]]
  return {'main_masks':masks,'aux_logits':torch.cat(aux,dim=1),'presence_logits':self.presence_head(states[:,:13].to(self.presence_head.weight.dtype)).squeeze(-1),'hidden':states,'logits':None,'semantic_guidance':False,'geometry':{'original_hw':f.orig_hw,'sam_input_hw':f.input_hw,'qwen_grid_thw':image_grid_thw[0].tolist(),'coordinate_convention':'full_frame_normalized_pixel_centers'}}
 def parameter_groups(self,learning_rates,weight_decay=.01):
  groups,counts=super().parameter_groups(learning_rates,weight_decay) if self.visual_queries is None else self._groups_with_queries(learning_rates,weight_decay)
  return groups,counts
 def _groups_with_queries(self,rates,wd):
  # Base MainlineModel expects its trainable set to equal its own groups.
  from .optimization import parameter_groups
  groups,counts=parameter_groups(self.base,rates,wd)
  for module in [self.presence_head,self.visual_queries]:
   for no_decay in [False,True]:
    ps=[p for n,p in module.named_parameters() if p.requires_grad and (p.ndim<2 or n.endswith('bias') or 'norm' in n or n=='queries')==no_decay]
    if ps:groups.append({'params':ps,'lr':rates['new'],'weight_decay':0. if no_decay else wd,'family':'new','no_decay':no_decay})
  allp=[p for g in groups for p in g['params']]
  require(len(allp)==len({id(p)for p in allp}) and {id(p)for p in allp}=={id(p)for p in self.parameters()if p.requires_grad},'optimizer complete unique inventory')
  counts['query_parameters']=sum(p.numel()for p in self.visual_queries.parameters())
  return groups,counts
