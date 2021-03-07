# DeepPET-OPSCC-Example
This contains codes for inference using trained DeepPET-OPSCC Models. Only need Segmentation mask and Original PET data as inputs. Default is Nifit data format (.nii.gz).

### Overview

<p align="center">
  <img align="center" src="Overview.png" width="640">
</p>


### Installation

This inference code is based on Pytorch and has been tested with the latest version (1.8.0)

- Ananoconda environment installation and activation
```
conda create -n deeppet-py3 pip python=3.6
conda activate deeppet-py3
```
- Install Pytorch and required packages
```
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c conda-forge
pip install SimpleITK
pip install tqdm
```
- Run the code, we put one patient example in sample_data folder.
```
python inference_DeepPETOPSCC.py
```


### Citation
If you find this repository useful in your research, please cite:
```
@article{Cheng2021,
  title={Deep learning-based fully-automated prediction of overall survival in patients with oropharyngeal cancer using FDG PET imaging: an international retrospective study.},
  author={Nai-Ming Cheng, Jiawen Yao, Jinzheng Cai, et al},
  journal={Under Review}
}

```
