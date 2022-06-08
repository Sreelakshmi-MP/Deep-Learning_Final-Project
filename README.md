# Enhancement of Auto-Segmentation Capabilities of Neural Networks 

 Abstract 
Auto-Segmentation automatically segments areas of interest in an image using self-supervised learning algorithms. DINO introduced the possibility of using Vision Transformers for Self-supervised learning. DINO has high accurate attention and fully unsupervised semantic segmentation capabilities. Trained ImageNet CNNs are strongly biased towards recognizing textures rather than shapes. On introducing Stylization of ImageNet, we observe that the existing measures of shape bias introduced due to 
Stylization may not necessarily be a stable predictor of generalization. We experimented different variations of the DINO architecture using ConvMixer as Student and Teacher networks. We do not observe significant improvement over the performance of traditional DINO architecture using Stylized ImageNet over ImageNet dataset.

You can evaluate the ViT with DINO as follows:
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --data_path <your data path> --output_dir <your output directory>
  
You can evaluate ConvMixer as follows:
python -m torch.distributed.launch --nproc_per_node=4 main_convmixer.py --data_path <your data path> --output_dir <your output directory>
  
