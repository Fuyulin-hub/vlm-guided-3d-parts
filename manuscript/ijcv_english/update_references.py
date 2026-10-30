"""Apply verified publication metadata without changing established citation keys."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
target = ROOT / 'references.bib'
text = target.read_text(encoding='utf-8')
updates = {
 'ravi2024sam2': ('inproceedings', {'year':'2025', 'booktitle':'International Conference on Learning Representations', 'url':'https://arxiv.org/abs/2408.00714'}),
 'xiang2025trellis2': ('inproceedings', {'year':'2026', 'booktitle':'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition', 'pages':'14419--14429', 'url':'https://openaccess.thecvf.com/content/CVPR2026/html/Xiang_Native_and_Compact_Structured_Latents_for_3D_Generation_CVPR_2026_paper.html'}),
 'ma2024find3d': ('inproceedings', {'year':'2025', 'booktitle':'Proceedings of the IEEE/CVF International Conference on Computer Vision', 'pages':'7818--7827', 'url':'https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Find_Any_Part_in_3D_ICCV_2025_paper.html'}),
 'liu2025partfield': ('inproceedings', {'title':'{PartField}: Learning {3D} Feature Fields for Part Segmentation and Beyond', 'year':'2025', 'booktitle':'Proceedings of the IEEE/CVF International Conference on Computer Vision', 'pages':'9704--9715', 'url':'https://openaccess.thecvf.com/content/ICCV2025/html/Liu_PartField_Learning_3D_Feature_Fields_for_Part_Segmentation_and_Beyond_ICCV_2025_paper.html'}),
 'zhu2025partsam': ('inproceedings', {'year':'2026', 'booktitle':'International Conference on Learning Representations', 'url':'https://proceedings.iclr.cc/paper_files/paper/2026/hash/d6be51e667e0b263e89a23294b57f8cf-Abstract-Conference.html'}),
 'deng2025geosam2': ('inproceedings', {'year':'2026', 'booktitle':'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition', 'pages':'6367--6376', 'url':'https://openaccess.thecvf.com/content/CVPR2026/html/Deng_GeoSAM2_Unleashing_the_Power_of_SAM2_for_3D_Part_Segmentation_CVPR_2026_paper.html'}),
 'li2026segvigen': ('article', {'year':'2026', 'journal':'ACM Transactions on Graphics', 'volume':'45', 'number':'4', 'pages':'1--12', 'note':'Article 68', 'doi':'10.1145/3811399', 'url':'https://doi.org/10.1145/3811399'}),
}
for key, (entry_type, fields) in updates.items():
    pattern = re.compile(r'@\w+\{' + re.escape(key) + r',\n.*?\n\}', re.S)
    match = pattern.search(text)
    if not match:
        raise ValueError(key)
    old = match.group(0)
    values = dict(re.findall(r'^\s*(\w+)\s*=\s*\{(.*)\},?\s*$', old, re.M))
    for field in ['howpublished', 'eprint', 'archivePrefix', 'primaryClass', 'doi', 'url']:
        values.pop(field, None)
    values.update(fields)
    replacement = '@' + entry_type + '{' + key + ',\n' + ',\n'.join('  '+k+' = {'+v+'}' for k,v in values.items()) + '\n}'
    text = text[:match.start()] + replacement + text[match.end():]

new_entries = r'''
@article{cen2025sa3d,
 author = {Cen, Jiazhong and Fang, Jiemin and Zhou, Zanwei and Yang, Chen and Xie, Lingxi and Zhang, Xiaopeng and Shen, Wei and Tian, Qi},
 title = {Segment Anything in {3D} with Radiance Fields},
 journal = {International Journal of Computer Vision},
 year = {2025}, volume = {133}, number = {8}, pages = {5138--5160},
 doi = {10.1007/s11263-025-02421-7}
}
@article{han2025dmpcl,
 author = {Han, Mengya and Luo, Yong and Hu, Han and Wang, Zengmao and Zhang, Lefei and Du, Bo and Duan, Ling-Yu and Tao, Dacheng},
 title = {{DM-PCL}: Text-Driven Dual-Modal Prototype Consistency Learning for Weakly-Supervised Few-Shot Part Segmentation},
 journal = {International Journal of Computer Vision},
 year = {2025}, volume = {133}, number = {11}, pages = {7553--7569},
 doi = {10.1007/s11263-025-02545-w}
}
@article{michieli2022gmenet,
 author = {Michieli, Umberto and Zanuttigh, Pietro},
 title = {Edge-Aware Graph Matching Network for Part-Based Semantic Segmentation},
 journal = {International Journal of Computer Vision},
 year = {2022}, volume = {130}, number = {11}, pages = {2797--2821},
 doi = {10.1007/s11263-022-01671-z}
}
@article{zhao2026cmpf,
 author = {Zhao, Sicheng and Chen, Xi and Yao, Hongxun and Yang, Haosen and Zhang, Yanhao and Jin, Sheng and Zhu, Xiatian and Lu, Haonan and Jiang, Kui and Ding, Guiguang},
 title = {{CMPF}: Harmonizing Cross-Model Prior Fusion for Open-Vocabulary Segmentation},
 journal = {International Journal of Computer Vision},
 year = {2026}, volume = {134}, number = {6}, pages = {Article 296},
 doi = {10.1007/s11263-026-02886-0}
}
@article{yang2024benchmark,
 author = {Yang, Yafei and Yang, Bo},
 title = {Benchmarking and Analysis of Unsupervised Object Segmentation from Real-World Single Images},
 journal = {International Journal of Computer Vision},
 year = {2024}, volume = {132}, number = {6}, pages = {2077--2113},
 doi = {10.1007/s11263-023-01973-w}
}
@article{gao2025latent,
 author = {Gao, Xiang and Hu, Wei and Liao, Renjie},
 title = {Learning Latent Part-Whole Hierarchies for Point Clouds},
 journal = {International Journal of Computer Vision},
 year = {2025}, volume = {133}, pages = {7312--7331},
 doi = {10.1007/s11263-025-02533-0}
}
@article{jain2019click,
 author = {Jain, Suyog Dutt and Grauman, Kristen},
 title = {Click Carving: Interactive Object Segmentation in Images and Videos with Point Clicks},
 journal = {International Journal of Computer Vision},
 year = {2019}, volume = {127}, pages = {1321--1344},
 doi = {10.1007/s11263-019-01184-2}
}
@article{li2025hierarchical,
 author = {Li, Chongshou and Liu, Yuheng and Li, Xinke and Zhang, Yuning and Li, Tianrui and Yuan, Junsong},
 title = {Deep Hierarchical Learning for {3D} Semantic Segmentation},
 journal = {International Journal of Computer Vision},
 year = {2025}, volume = {133}, pages = {4420--4441},
 doi = {10.1007/s11263-025-02387-6}
}
@article{ding2026grex,
 author = {Ding, Henghui and Liu, Chang and He, Shuting and Jiang, Xudong and Jiang, Yu-Gang},
 title = {{GREx}: Generalized Referring Expression Segmentation, Comprehension, and Generation},
 journal = {International Journal of Computer Vision},
 year = {2026}, volume = {134}, pages = {Article 79},
 doi = {10.1007/s11263-026-02729-y}
}
@article{mei2024clurender,
 author = {Mei, Guofeng and Saltori, Cristiano and Ricci, Elisa and Sebe, Nicu and Wu, Qiang and Zhang, Jian and Poiesi, Fabio},
 title = {Unsupervised Point Cloud Representation Learning by Clustering and Neural Rendering},
 journal = {International Journal of Computer Vision},
 year = {2024}, volume = {132}, pages = {3251--3269},
 doi = {10.1007/s11263-024-02027-5}
}
@misc{dang2025decoupled,
 author = {Dang, Jisheng and Wu, Xudong and Wang, Bimei and Lv, Ning and Chen, Jiayu and Zhao, Jingwen and Liu, Yichu and Liu, Jizhao and Li, Juncheng and Wang, Teng},
 title = {Decoupled Seg Tokens Make Stronger Reasoning Video Segmenter and Grounder},
 year = {2025}, howpublished = {arXiv preprint arXiv:2506.22880},
 doi = {10.48550/arXiv.2506.22880}, url = {https://arxiv.org/abs/2506.22880}
}
'''
if 'cen2025sa3d,' not in text:
    text += '\n' + new_entries
def compact_identifiers(match):
    block = match.group(0)
    has_doi = bool(re.search(r'\bdoi\s*=', block))
    print_proceedings = block.startswith('@inproceedings')
    if has_doi or print_proceedings:
        block = re.sub(r'^\s*url\s*=\s*\{[^\n]*\},?\s*\n', '', block, flags=re.M)
    block = re.sub(r'\\href\{[^}]+\}\{([^}]+)\}', r'\1', block)
    def protect_venue(venue):
        value = venue.group(2)
        if not value.startswith('{'):
            value = '{' + value + '}'
        return venue.group(1) + value + venue.group(3)
    block = re.sub(r'^(\s*booktitle\s*=\s*\{)(.*)(\},?\s*)$', protect_venue, block, flags=re.M)
    return block
text = re.sub(r'@\w+\{[^\n]+,\n.*?\n\}', compact_identifiers, text, flags=re.S)
target.write_text(text, encoding='utf-8')
print('Verified publication entries updated; supplied IJCV reading entries retained.')
