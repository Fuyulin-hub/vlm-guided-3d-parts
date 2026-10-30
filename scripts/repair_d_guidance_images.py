"""Repair supplied D guidance color encodings; inferred regions remain review candidates."""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from recolor_d_guidance_to_bc_palette import recolor_labels
from prepare_d_repair_review import thumb


def foreground_mask(rgba):
    rgb=rgba[...,:3].astype(np.int16)
    # The supplied palette has no white/gray foreground. Bright neutral pixels
    # here are white haze or baked checkerboard; dark neutral cap/bands survive.
    bright_neutral=(rgb.min(axis=2)>=205)&((rgb.max(axis=2)-rgb.min(axis=2))<=50)
    return (rgba[...,3]>=128)&~bright_neutral


def nearest_rgb(pixels, centers):
    centers=np.asarray(centers,dtype=np.float32)
    result=np.empty(len(pixels),dtype=np.int32)
    distances=np.empty(len(pixels),dtype=np.float32)
    for start in range(0,len(pixels),20000):
        p=pixels[start:start+20000].astype(np.float32)
        d=((p[:,None,:]-centers[None,:,:])**2).sum(axis=2)
        ids=d.argmin(axis=1)
        result[start:start+len(p)]=ids
        distances[start:start+len(p)]=np.sqrt(d[np.arange(len(p)),ids])
    return result,distances


def infer_regions(rgba, source_palette, stable_color_seeds=False):
    fg=foreground_mask(rgba)
    if not np.any(fg):
        raise ValueError("empty foreground after background cleanup")
    rgb=rgba[...,:3]
    pixels=rgb[fg]
    labels=np.zeros(fg.shape,dtype=np.uint16)
    packed=(pixels[:,0].astype(np.uint32)<<16)|(pixels[:,1].astype(np.uint32)<<8)|pixels[:,2]
    palette_counts={label_id:int(np.sum(packed==((c[0]<<16)|(c[1]<<8)|c[2]))) for label_id,c in source_palette.items()}
    exact_fraction=sum(palette_counts.values())/len(pixels)
    info={"source_exact_palette_fraction":float(exact_fraction),
          "source_foreground_pixels":int(fg.sum()),"alpha_threshold":128,
          "removed_bright_neutral_pixels":int(((rgba[...,3]>=128)&~fg).sum())}
    if exact_fraction>=0.85:
        active=[k for k,n in palette_counts.items() if n>=max(3,int(len(pixels)*0.00003))]
        centers=[source_palette[k] for k in active]
        indices,distances=nearest_rgb(pixels,centers)
        labels[fg]=np.asarray(active,dtype=np.uint16)[indices]
        colors={k:source_palette[k] for k in active}
        info.update(method="active_source_palette",uncertain_fraction=float(np.mean(distances>50)))
        return labels,colors,info

    hsv=np.asarray(Image.fromarray(rgb).convert("HSV"))
    hue=hsv[...,0].astype(np.float32)
    saturation=hsv[...,1].astype(np.float32)/255
    value=hsv[...,2].astype(np.float32)/255
    neutral=fg&(saturation<0.20)&(value<0.78)
    eroded=np.asarray(Image.fromarray((neutral*255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)))>0
    use_neutral=eroded.sum()>=max(4,int(fg.sum()*0.00005))
    core=fg&(saturation>=0.28)&(value>=0.12)&(rgba[...,3]>=200)
    if core.sum()<8:
        core=fg&(saturation>=0.20)
    if stable_color_seeds:
        # Only for the nine low-resolution solid screenshots. Interior seeds
        # avoid assigning new IDs to color mixtures along antialiased borders.
        # Do not apply this erosion to fine thin structures.
        h8=hsv[...,0]
        def span(values):
            im=Image.fromarray(values)
            return np.asarray(im.filter(ImageFilter.MaxFilter(3))).astype(int)-np.asarray(im.filter(ImageFilter.MinFilter(3))).astype(int)
        stable=np.minimum(span(h8),span(((h8.astype(int)+128)%256).astype(np.uint8)))<=5
        inside=np.asarray(Image.fromarray((fg*255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)))>0
        stable_core=core&stable&inside
        if stable_core.sum()>=8:
            core=stable_core
        info['stable_interior_color_seeds']=int(core.sum())
    hist=np.bincount(hue[core].astype(np.int32),weights=saturation[core]**2,minlength=256)
    axis=np.arange(-6,7)
    kernel=np.exp(-0.5*(axis/1.5)**2)
    kernel/=kernel.sum()
    smooth=np.convolve(np.pad(hist,(6,6),mode="wrap"),kernel,mode="valid")
    candidates=np.flatnonzero((smooth>=np.roll(smooth,1))&(smooth>=np.roll(smooth,-1))&(smooth>0))
    peaks=[]
    minimum_mass=max(8,float(core.sum())*0.0007)
    for peak in sorted(candidates,key=lambda p:-smooth[p]):
        mass=sum(hist[(int(peak)+j)%256] for j in range(-5,6))
        if mass<minimum_mass:
            continue
        if all(min(abs(int(peak)-p),256-abs(int(peak)-p))>=8 for p in peaks):
            peaks.append(int(peak))
    if not peaks and core.any():
        peaks=[int(np.argmax(smooth))]
    peaks=sorted(peaks)
    if not peaks and not use_neutral:
        raise ValueError("no stable foreground color")
    if peaks:
        delta=np.abs(hue[fg,None]-np.asarray(peaks)[None,:])
        delta=np.minimum(delta,256-delta)
        assigned=delta.argmin(axis=1)+1
        labels[fg]=assigned
        uncertainty=float(np.mean(delta.min(axis=1)>10))
    else:
        labels[fg]=1
        uncertainty=0.0
    if use_neutral:
        labels[neutral]=len(peaks)+1
    colors={}
    for label_id in np.unique(labels):
        if label_id:
            part=(labels==label_id)
            stable=part&(rgba[...,3]>=220)
            colors[int(label_id)]=tuple(int(x) for x in np.median(rgb[stable if stable.any() else part],axis=0))
    # Low-saturation edge mixtures should follow an existing source color,
    # while genuine dark neutral regions retain their separate ID.
    edges=fg&(saturation<0.28)
    if use_neutral:
        edges &= ~neutral
    if edges.any():
        keys=list(colors)
        nearest,_=nearest_rgb(rgb[edges],[colors[k] for k in keys])
        labels[edges]=np.asarray(keys,dtype=np.uint16)[nearest]
    present=set(int(x) for x in np.unique(labels) if x)
    colors={k:v for k,v in colors.items() if k in present}
    info.update(method="dominant_hue_regions",hue_peak_degrees=[round(x*360/256,2) for x in peaks],
                neutral_region=bool(use_neutral),uncertain_fraction=uncertainty)
    return labels,colors,info


def bbox(mask):
    y,x=np.nonzero(mask)
    if not len(x):
        raise ValueError("empty silhouette")
    return np.array([x.min(),y.min(),x.max()+1,y.max()+1],dtype=float)


def align_to_raw(labels,raw_mask):
    source_box=bbox(labels>0)
    target_box=bbox(raw_mask)
    source_extent=source_box[2:]-source_box[:2]
    target_extent=target_box[2:]-target_box[:2]
    scale=float(np.dot(source_extent,target_extent)/np.dot(source_extent,source_extent))
    source_center=(source_box[:2]+source_box[2:])/2
    target_center=(target_box[:2]+target_box[2:])/2
    tx,ty=(target_center-scale*source_center).tolist()
    image=Image.fromarray(labels).transform((raw_mask.shape[1],raw_mask.shape[0]),Image.Transform.AFFINE,
        (1/scale,0,-tx/scale,0,1/scale,-ty/scale),resample=Image.Resampling.NEAREST,fillcolor=0)
    return np.asarray(image),{"scale":scale,"translation_xy":[tx,ty],"rotation_degrees":0,
                            "flip":False,"source_bbox":source_box.tolist(),"target_bbox":target_box.tolist()}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    rows=json.loads((args.source/"source_manifest.json").read_text(encoding="utf-8"))
    palette_path=root/"downloads/part_palette_30_high_contrast_20261021.json"
    palette={int(e['label_id']):tuple(e['rgb']) for e in json.loads(palette_path.read_text(encoding="utf-8"))['colors']}
    bc_path=root/"downloads/colors_pure_bc_20261022.json"
    bc=json.loads(bc_path.read_text(encoding="utf-8"))
    target=[tuple(c[:3]) for c in bc[1:]]
    strict_root=root/"downloads/D_solid_bc20_recolored_20261023"
    strict=json.loads((strict_root/"recolor_receipt.json").read_text(encoding="utf-8"))['per_uid']
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    shutil.copyfile(bc_path,args.output/"colors_pure_bc.json")
    shutil.copyfile(palette_path,args.output/"source_palette_30.json")
    records=[]
    for i,row in enumerate(rows,1):
        uid,group=row['uid'],row['group']
        human_path=args.source/row['human_path']
        raw_path=args.source/row['raw_path']
        with Image.open(human_path) as image: rgba=np.asarray(image.convert('RGBA'))
        with Image.open(raw_path) as image: raw_rgba=np.asarray(image.convert('RGBA'))
        raw_mask=raw_rgba[...,3]>=128
        guide_path=args.output/"guidance"/"D_human_fixed_bc20"/group/f"{uid}.png"
        label_path=args.output/"labels"/group/uid/"labels_uint16.png"
        guide_path.parent.mkdir(parents=True,exist_ok=True)
        label_path.parent.mkdir(parents=True,exist_ok=True)
        flags=[]
        if uid in strict:
            labels_source=strict_root/"references"/group/uid/"labels_uint16.png"
            shutil.copyfile(labels_source,label_path)
            shutil.copyfile(strict_root/"guidance"/"D_human_fixed_bc20"/group/f"{uid}.png",guide_path)
            with Image.open(label_path) as image: labels=np.asarray(image)
            info={"method":"previous_exact_conversion","status":"exact_encoding_semantics_unreviewed"}
            mapping={int(k):v for k,v in strict[uid]['label_to_bc_palette_index'].items()}
        else:
            small_solid=group=='solid_volume' and rgba.shape[:2]!=raw_mask.shape
            labels,colors,info=infer_regions(rgba,palette,stable_color_seeds=small_solid)
            if uid=='6c6279ff246e4d3fb46d609fb3faaaa1':
                # Visually checked source: red top, orange rim, shaded yellow
                # wall, and shaded dark hoops. Consolidate shades within each.
                shade_map={}
                for old_id,hue in enumerate(info['hue_peak_degrees'],1):
                    shade_map[old_id]=3 if 35<=hue<=90 else (4 if 150<=hue<=250 else (2 if 20<=hue<35 else 1))
                if info.get('neutral_region'):
                    shade_map[len(info['hue_peak_degrees'])+1]=4
                merged=np.zeros_like(labels)
                for old_id,new_id in shade_map.items(): merged[labels==old_id]=new_id
                labels=merged
                colors={int(k):tuple(int(v) for v in np.median(rgba[...,:3][labels==k],axis=0)) for k in np.unique(labels) if k}
                info['inspected_shade_merge']={'source_candidate_to_final_id':shade_map,'regions':['top','rim','wall','hoops']}
            info['status']='automatic_repair_needs_review'
            if labels.shape!=raw_mask.shape:
                labels,transform=align_to_raw(labels,raw_mask)
                info['source_to_raw_frame']=transform
                flags.append('rescaled_from_small_source')
            guide,mapping=recolor_labels(labels,colors,target)
            Image.fromarray(labels.astype(np.uint16)).save(label_path)
            Image.fromarray(guide).save(guide_path)
            if info['method']=='dominant_hue_regions': flags.append('inferred_color_regions')
            if human_path.suffix.lower() in {'.jpg','.jpeg'}: flags.append('jpeg_source')
            if info['uncertain_fraction']>0.02: flags.append('color_ambiguity')
        mask=labels>0
        iou=float(np.sum(mask&raw_mask)/max(1,np.sum(mask|raw_mask)))
        if iou<0.85: flags.append('silhouette_differs_from_raw')
        info.update(uid=uid,group=group,source_archive=row['archive'],human_sha256=sha(human_path),
                    source_size=row['size'],raw_silhouette_iou=iou,review_flags=flags,
                    label_count=len(mapping),color_reuse=len(set(mapping.values()))<len(mapping),
                    label_to_bc_palette_index={str(k):v for k,v in mapping.items()},
                    guide_sha256=sha(guide_path),labels_sha256=sha(label_path))
        source_dir=args.output/"sources"/group
        source_dir.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(human_path,source_dir/human_path.name)
        shutil.copyfile(raw_path,source_dir/raw_path.name)
        info['human_file']=str((source_dir/human_path.name).relative_to(args.output)).replace('\\','/')
        info['raw_file']=str((source_dir/raw_path.name).relative_to(args.output)).replace('\\','/')
        info['guide_file']=str(guide_path.relative_to(args.output)).replace('\\','/')
        info['labels_file']=str(label_path.relative_to(args.output)).replace('\\','/')
        records.append(info)
        print(f"{i}/{len(rows)} {uid[:8]} parts={len(mapping)} {info['method']} IoU={iou:.3f}",flush=True)
    receipt={'total':len(records),'exact':sum(x['method']=='previous_exact_conversion' for x in records),
             'repaired':sum(x['method']!='previous_exact_conversion' for x in records),
             'palette_sha256':sha(bc_path),'per_uid':records}
    (args.output/'repair_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (args.output/'review_list.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['group','uid','status','method','region_count','silhouette_iou','review_flags'])
        for item in records: writer.writerow([item['group'],item['uid'],item['status'],item['method'],item['label_count'],round(item['raw_silhouette_iou'],4),'|'.join(item['review_flags'])])
    make_review(args.output,records)
    print(json.dumps({k:v for k,v in receipt.items() if k!='per_uid'}))


def make_review(output,records):
    review_dir=output/'review'
    review_dir.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    cards=[]
    for group in ['thin_fine_grained','solid_volume']:
        group_rows=[x for x in records if x['group']==group]
        for page,start in enumerate(range(0,len(group_rows),10),1):
            sheet=Image.new('RGB',(1000,1150),'white')
            draw=ImageDraw.Draw(sheet)
            for slot,row in enumerate(group_rows[start:start+10]):
                x,y=(slot%2)*500,(slot//2)*230
                draw.text((x+5,y+3),f"{row['uid'][:12]} / {row['label_count']} colors / IoU {row['raw_silhouette_iou']:.2f}",fill='black',font=font)
                for j,(key,name) in enumerate([('raw_file','raw'),('human_file','D source'),('guide_file','repaired')]):
                    draw.text((x+5+j*165,y+23),name,fill='black',font=font)
                    with Image.open(output/row[key]) as im: sheet.paste(thumb(im),(x+5+j*165,y+45))
            sheet.save(review_dir/f'{group}_{page:02}.jpg',quality=95)
    for row in records:
        images=''.join(f'<figure><a href="{html.escape(row[key])}"><img loading="lazy" src="{html.escape(row[key])}"></a><figcaption>{caption}</figcaption></figure>' for key,caption in [('raw_file','raw 输入'),('human_file','原 D 人工图'),('guide_file','改色结果')])
        cards.append(f'<article data-group="{row["group"]}"><h2>{row["group"]} / {row["uid"]}</h2><p>{row["method"]} · {row["label_count"]} 个颜色区域 · 轮廓 IoU={row["raw_silhouette_iou"]:.3f} · {html.escape(", ".join(row["review_flags"]))}</p><div>{images}</div><a href="{row["labels_file"]}">整数标签</a></article>')
    document='''<!doctype html><meta charset="utf-8"><title>D 引导图改色核对</title><style>body{font:16px system-ui;margin:30px;background:#f3f4f6;color:#18212b}h1{font-size:26px}h2{font-size:17px}article{background:white;padding:20px;margin:20px 0;border-radius:12px}article>div{display:flex;gap:18px}figure{margin:0;width:31%}img{width:100%;background:repeating-conic-gradient(#e4e4e4 0% 25%,#fafafa 0% 50%) 0/24px 24px}figcaption{padding:8px}p{line-height:1.6}a{color:#1564c0}</style><h1>D 引导图改色：90 张三图核对</h1><p>15 张保持原严格转换结果；75 张为自动修复候选。背景已透明、前景为 B/C 色板纯色。请核对视角、轮廓和部件边界，再决定是否用于评估。轮廓 IoU 仅衡量与 raw 前景的几何重合，不衡量分割准确率。</p>'''+''.join(cards)
    (output/'index.html').write_text(document,encoding='utf-8')
    priority=[card for row,card in zip(records,cards) if 'silhouette_differs_from_raw' in row['review_flags']]
    head=document.split('<h1>',1)[0]
    (output/'priority_review.html').write_text(head+'<h1>优先核对：原 D 与 raw 轮廓差异较大的 5 张</h1><p>颜色规范化已完成。以下源人工图的轮廓与 raw 差异较大，需核对源标注的边界和对应关系，再纳入评估。</p>'+''.join(priority),encoding='utf-8')

if __name__=='__main__':
    main()
