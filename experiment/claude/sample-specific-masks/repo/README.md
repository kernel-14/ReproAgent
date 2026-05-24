# Sample-specific Masks for Visual Reprogramming-based Prompting

Code reproduction for **“Sample-specific Masks for Visual Reprogramming-based Prompting”**. The repository implements the paper’s visual reprogramming route around sample-specific multi-channel masks (SMM): a frozen ImageNet-1K pretrained classifier is adapted to a target classification task by learning a shared input-space pattern `δ` and a lightweight CNN mask generator `f_mask`, while output mapping remains non-parametric.

reference_grounding: chunk_009 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md  
reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md  
reference_grounding: chunk_014_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md  
reference_grounding: chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md  
reference_grounding: chunk_002_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md

## Canonical route

Default smoke validation executes the same data/model/method/train/evaluate/artifact route as full mode, with bounded samples and batches: