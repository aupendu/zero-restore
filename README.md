# Zero-shot Single Image Restoration through Controlled Perturbation of Koschmieder's Model
This is the PyTorch implementation for our CVPR 2021 paper:

**Aupendu Kar, Sobhan Kanti Dhara, Debashis Sen, Prabir Kumar Biswas. Zero-shot Single Image Restoration through Controlled Perturbation of Koschmieder's Model. [[Project Website]](https://aupendu.github.io/zero-restore) [[PAPER]](https://openaccess.thecvf.com/content/CVPR2021/html/Kar_Zero-Shot_Single_Image_Restoration_Through_Controlled_Perturbation_of_Koschmieders_Model_CVPR_2021_paper.html)

## Dependencies
* Python 3.6
* imageio==2.6.1
* numpy==1.17.4
* torch==1.8.0+cu111
* torchvision==0.9.0+cu111
* tqdm==4.40.2

## Test Datasets and Results of Our Algorithm
All the processed datasets that are used in this paper and the output images of the proposed algorithm are given below: 
1. Image Dehazing (Table 1 of Main Paper) [Google Drive](https://drive.google.com/drive/folders/1Da1AHZNm0lfrpUYS9Ag1SIQ6QXkFdmHV?usp=sharing)
2. Underwater Image Enhancement (Table 2 of Main Paper) [Google Drive](https://drive.google.com/drive/folders/1EC5ui1sDnc4049CBpjZyz5WrMeTNyaqA?usp=sharing)
3. Lowlight Image Enhancement (Table 3 of Main Paper) [Google Drive](https://drive.google.com/drive/folders/196El-g_riGefDcrlT4AXx8biZ9n4WTHE?usp=sharing)

## Test Codes
1. Run ``HazeZeroShot.py`` to perform single image dehazing
2. Run ``UnderWaterZeroShot.py`` to perform underwater image enhancement
3. Run ``LowLightZeroShot.py`` to perform lowlight image enhancement

* Give test image set path through ``--TestFolderPath`` argument
* Give the path of results through ``--SavePath`` argument

## Citation
```
@InProceedings{Kar_2021_CVPR,
  author = {Kar, Aupendu and Dhara, Sobhan Kanti and Sen, Debashis and Biswas, Prabir Kumar},
  title = {Zero-shot Single Image Restoration through Controlled Perturbation of Koschmieder’s Model},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2021},
  pages = {16205-16215}
}
```

## Contact
Aupendu Kar: mailtoaupendu[at]gmail[dot]com
