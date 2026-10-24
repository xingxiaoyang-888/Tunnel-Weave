"""CPU scoring contract. No predictions or measurements fabricated here."""
from collections import defaultdict
import numpy as np
from .contracts import CLASSES,ContractError
from .data import validate_bundle


def _pixels(pred,target,valid):
    tp=int((pred&target&valid).sum());fp=int((pred&~target&valid).sum());fn=int((~pred&target&valid).sum())
    union=tp+fp+fn;den=2*tp+fp+fn
    return {'tp_pixels':tp,'fp_pixels':fp,'fn_pixels':fn,'valid_predicted_pixels':tp+fp,
            'iou':tp/union if union else None,'dice':2*tp/den if den else None,'overlap_hit':tp>0}


def score_sample(result,bundle,row):
    validate_bundle(bundle,row)
    records=result['classes']
    if len(records)!=13 or [x['class_name'] for x in records]!=CLASSES:
        raise ContractError('prediction channel order mismatch')
    full=set(row['full_image_coverage_classes']);rows=[]
    for i,(c,x) in enumerate(zip(CLASSES,records)):
        target=np.asarray(bundle['target'][i],dtype=bool);valid=np.asarray(bundle['valid'][i],dtype=bool)
        raw=np.asarray(x['raw_mask']);gated=np.asarray(x['gated_mask'])
        if raw.dtype!=np.bool_ or gated.dtype!=np.bool_ or raw.shape!=target.shape or gated.shape!=target.shape:
            raise ContractError('prediction/GT coordinate mismatch')
        expected=raw if x['predicted_present'] else np.zeros_like(raw)
        if not np.array_equal(gated,expected):raise ContractError('saved gating inconsistent with presence decision')
        state=int(bundle['presence'][i]);positive=bool((target&valid).any());roi=c in row['region_coverage_classes']
        a=_pixels(raw,target,valid);b=_pixels(gated,target,valid)
        rows.append({'sample_id':row['sample_id'],'class_name':c,'source':row.get('source_dataset','unknown'),
            'parent_group':row.get('group_id','unknown'),'coverage':'full_image' if c in full else 'ROI' if roi else 'unknown',
            'presence_truth':state,'presence_predicted':bool(x['predicted_present']),
            'raw_mask_detected':bool(raw.any()),'gated_mask_detected':bool(gated.any()),
            'raw_mask_detected_in_valid_region':bool((raw&valid).any()),'gated_mask_detected_in_valid_region':bool((gated&valid).any()),
            'known_pixel_positive':positive,'raw':a,'gated':b,
            'positive_iou_including_miss':float(b['iou']) if positive else None,
            'positive_dice_including_miss':float(b['dice']) if positive else None,
            'raw_query_positive_iou':float(a['iou']) if positive else None,
            'raw_query_is_not_GT_conditioned_diagnostic':True,
            'roi_zero_target':roi and valid.any().item() and not positive,
            'full_image_confirmed_negative':state==0,
            'full13_normal':bool(row.get('full_normal_for_13',False)),
            'diseased_image':bool(row['positive_classes'])})
    return rows


def aggregate(rows):
    def mean(v):return sum(v)/len(v) if v else None
    per_class={}
    for c in CLASSES:
        rs=[x for x in rows if x['class_name']==c and x['coverage']=='full_image' and x['presence_truth']!=-1]
        support=sum(x['presence_truth']==1 for x in rs);d={}
        for channel,key in [('presence','presence_predicted'),('gated_mask','gated_mask_detected')]:
            tp=sum(x['presence_truth']==1 and x[key] for x in rs);fn=support-tp;fp=sum(x['presence_truth']==0 and x[key] for x in rs)
            d[channel]={'tp':tp,'fn':fn,'fp':fp,'positive_support':support,
                'recall':tp/support if support else None,'f1':2*tp/(2*tp+fp+fn) if support else None}
        per_class[c]=d
    positives=[x for x in rows if x['known_pixel_positive']]
    roi_zero=[x for x in rows if x['roi_zero_target']]
    disease_negative=[x for x in rows if x['diseased_image'] and x['full_image_confirmed_negative']]
    normal={x['sample_id'] for x in rows if x['full13_normal']}
    normal_false={x['sample_id'] for x in rows if x['full13_normal'] and (x['presence_predicted'] or x['gated_mask_detected'])}
    return {'per_class_full_image':per_class,
       'macro_presence_f1':mean([v['presence']['f1'] for v in per_class.values() if v['presence']['f1'] is not None]),
       'macro_gated_mask_f1':mean([v['gated_mask']['f1'] for v in per_class.values() if v['gated_mask']['f1'] is not None]),
       'macro_supported_class_count':sum(v['presence']['positive_support']>0 for v in per_class.values()),
       'known_positive_pairs':len(positives),'positive_iou_miss_zero':mean([x['positive_iou_including_miss'] for x in positives]),
       'positive_dice_miss_zero':mean([x['positive_dice_including_miss'] for x in positives]),
       'raw_query_positive_iou':mean([x['raw_query_positive_iou'] for x in positives]),
       'by_coverage':{coverage:{'positive_pairs':sum(x['coverage']==coverage for x in positives),
            'iou_miss_zero':mean([x['positive_iou_including_miss'] for x in positives if x['coverage']==coverage])} for coverage in ['full_image','ROI']},
       'normal_images':len(normal),'normal_images_with_any_presence_or_gated_mask_false_positive':len(normal_false),
       'diseased_image_other_known_class_FP':sum(x['presence_predicted'] for x in disease_negative),
       'diseased_image_other_known_class_gated_mask_FP':sum(x['gated_mask_detected'] for x in disease_negative),
       'ROI_zero_pairs':len(roi_zero),'ROI_zero_nonempty_gated_masks':sum(x['gated_mask_detected_in_valid_region'] for x in roi_zero),
       'ROI_zero_false_positive_pixels':sum(x['gated']['fp_pixels'] for x in roi_zero),
       'description_factuality':None,'unknown_detection_performance':None}
