"""Synthetic CPU contract tests, not actual model performance/learning evidence."""


from pathlib import Path


import inspect


import contextlib


import json


import tempfile


import types


import unittest


from unittest.mock import patch


import torch


from torch import nn


import torch.nn.functional as F


import tunnelweave.backbone as base_module


from tunnelweave.contracts import ContractError


from tunnelweave.model import MainlineModel


class FakeLanguage(nn.Module):
    def __init__(self):
        super().__init__();self.lora_A=nn.Parameter(torch.eye(4)*.02)
    def forward(self,inputs_embeds,**kw):
        state=inputs_embeds.cumsum(1)
        return types.SimpleNamespace(last_hidden_state=state+F.linear(state,self.lora_A))


class FakeQBase(nn.Module):
    def __init__(self):
        super().__init__();self.visual=nn.Linear(4,4);self.visual.spatial_merge_size=2
        self.language_model=FakeLanguage();self.config=types.SimpleNamespace(image_token_id=4)
    def get_image_features(self,pixel_values,image_grid_thw,**kw):
        return types.SimpleNamespace(pooler_output=(self.visual(pixel_values),))
    def get_placeholder_mask(self,ids,embeddings,image_features):
        return (ids==4)[...,None].expand_as(embeddings),None
    def compute_3d_position_ids(self,input_ids,**kw):
        return torch.arange(input_ids.shape[1]).reshape(1,1,-1).expand(3,1,-1)


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Embedding(16,4);self.lm_head=nn.Linear(4,16)
        self.model=FakeQBase();self.config=types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=4))
    def get_input_embeddings(self):return self.embed


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.full((1,4),.1));self.conv_s0=nn.Linear(4,4);self.conv_s1=nn.Linear(4,4)


class FakeSAM(nn.Module):
    def __init__(self):
        super().__init__();self.hidden_dim=4;self.sam_prompt_embed_dim=4
        self.image_encoder=nn.Linear(4,4);self.sam_mask_decoder=FakeDecoder()


class FakeInteractive:
    def __init__(self,sam):self.model=sam
    def encode_once(self,image):
        with torch.no_grad():enc=self.model.image_encoder(image)
        high=self.model.sam_mask_decoder.conv_s0(enc)+self.model.sam_mask_decoder.conv_s1(enc)
        return types.SimpleNamespace(image_embed=enc.T.reshape(1,4,2,2),high_res_feats=[high],orig_hw=(4,4),input_hw=(1024,1024))
    def decode(self,features,prompt_embedding,decoder=None):
        dec=decoder or self.model.sam_mask_decoder
        value=F.linear(prompt_embedding,dec.weight).reshape(1,1,1,1)+features.high_res_feats[0].mean()*.01
        return value.expand(1,1,4,4)


def make_base():
    torch.manual_seed(17)
    with patch.object(base_module,'Sam21Interactive',FakeInteractive):
        return base_module.TunnelScopeModel(FakeQwen(),FakeSAM(),1,2,3,cross_inner_dim=8,cross_heads=2,fusion_mode='sequence_concat')


def inputs():
    return dict(image=torch.ones(4,4),input_ids=torch.tensor([[4]*4+[2,3]+[1]*13]),pixel_values=torch.ones(4,4),image_grid_thw=torch.tensor([[1,4,4]]))


class MainlineTests(unittest.TestCase):
    def test_returns_13_own_seg_scores_and_13_masks(self):
        model=MainlineModel(make_base());out=model(**inputs())
        self.assertEqual(tuple(out['presence_logits'].shape),(1,13));self.assertEqual(len(out['main_masks']),13)
        states=base_module.own_token_state(out['hidden'],inputs()['input_ids'],1)
        self.assertTrue(torch.equal(out['presence_logits'],model.presence_head(states).squeeze(-1)))
    def test_presence_and_all_mask_gradients_reachable_without_gate(self):
        model=MainlineModel(make_base());model.train()
        with torch.no_grad():model.presence_head.weight.zero_();model.presence_head.bias.fill_(-100)
        out=model(**inputs())
        self.assertTrue(bool((out['presence_logits']< -99).all()))
        for mask in out['main_masks']:mask.retain_grad()
        loss=F.binary_cross_entropy_with_logits(out['presence_logits'],torch.ones_like(out['presence_logits']))
        loss=loss+sum((mask-1).square().mean() for mask in out['main_masks'])/13+out['aux_logits'].square().mean()*.2
        loss.backward()
        self.assertTrue(all(mask.grad is not None and bool((mask.grad!=0).any()) for mask in out['main_masks']))
        for module in [model.presence_head,model.base.main_projection,model.base.sam_model.sam_mask_decoder,model.base.fusion,model.base.aux_projection,model.base.aux_decoder]:
            self.assertTrue(any(p.grad is not None and bool((p.grad!=0).any()) for p in module.parameters()))
        self.assertTrue(any(p.grad is not None and bool((p.grad!=0).any()) for n,p in model.base.qwen.named_parameters() if 'lora_' in n))
    def test_forward_has_no_gt_or_metadata_parameters(self):
        self.assertEqual(list(inspect.signature(MainlineModel.forward).parameters),['self','image','input_ids','pixel_values','image_grid_thw','attention_mask'])
    def test_wrong_seg_count_rejected(self):
        model=MainlineModel(make_base());args=inputs();args['input_ids']=args['input_ids'][:,:-1]
        with self.assertRaisesRegex(ContractError,'13 SEG'):model(**args)
    def test_missing_aux_anchor_rejected(self):
        model=MainlineModel(make_base());args=inputs();args['input_ids'][0,4]=5
        with self.assertRaisesRegex(ContractError,'1 NOR'):model(**args)
    def test_unpadded_one_sample_guard(self):
        model=MainlineModel(make_base());args=inputs();args['input_ids']=args['input_ids'].repeat(2,1)
        with self.assertRaises(ContractError):model(**args)
    def test_frozen_guards_and_stage_b_transition(self):
        model=MainlineModel(make_base());model.set_training_stage('frozen')
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        with self.assertRaises(ContractError):model.train()
        with self.assertRaises(ContractError):model.set_training_stage('A')
        model.set_training_stage('B');model.train();self.assertTrue(model.presence_head.weight.requires_grad)
    def test_visual_encoders_stay_eval(self):
        model=MainlineModel(make_base()).train()
        self.assertFalse(model.base.qwen.model.visual.training);self.assertFalse(model.base.sam_model.image_encoder.training)
        self.assertFalse(any(p.requires_grad for p in model.base.qwen.model.visual.parameters()))
    def test_optimizer_groups_complete_head_in_new(self):
        model=MainlineModel(make_base());groups,counts=model.parameter_groups({'new':2e-5,'lora':1e-5,'decoder':1e-6})
        found=[g for g in groups if any(p is model.presence_head.weight for p in g['params'])]
        self.assertEqual(len(found),1);self.assertEqual(found[0]['family'],'new');self.assertEqual(found[0]['lr'],2e-5)
        ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})




if __name__=='__main__':unittest.main()
