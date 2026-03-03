

## Dataset Preparation

1. Download all 11 prompt-tuning datasets **with foreground views** on [[🤗HuggingFace](https://huggingface.co/datasets/JREion/Prompt_Tuning_Datasets_with_Foreground)]. <br>
  (This dataset is fully compatible with other prompt tuning methods that do not use foreground views!) <br>
  If you need a separate mask directory that **_does not contain raw images_**, please download it from one of the following links: [Google Drive] or [BaiduYun]
3. We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. <br>
  The file structure should be organized as:

    - **ImageNet**:
       (Please first prepare raw ImageNet dataset from: [[Raw Images](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description)] [[annoations](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing)] [[val conversion script](https://github.com/soumith/imagenetloader.torch/blob/master/valprep.sh)] )
          
          $DATA/
          |–– imagenet-1k/
          |   |–– imagenet/
          |   |   |-- images/
          |   |   |   |-- train/
          |   |   |   |   |-- n01440764/
          |   |   |   |   |-- n01443537/
          |   |   |   |   |-- ...
          |   |   |   |-- val/
          |   |   |   |   |-- n01440764/
          |   |   |   |   |-- n01443537/
          |   |   |   |   |-- ...
          |   |   |   |-- test/
          |   |   |   |   |-- ILSVRC2012_test_00000001.JPEG
          |   |   |   |   |-- ILSVRC2012_test_00000002.JPEG
          |   |   |   |   |-- ...
          |   |   |–– classnames.txt
          |   |   |–– command.txt
          |   |–– mask/

    - **Caltech101**:
   
          $DATA/
          |–– caltech-101/
          |   |–– caltech-101/
          |   |   |-- 101_ObjectCategories/
          |   |   |-- Annoations/
          |   |   |-- ...
          |   |–– mask/

   - **Food101**:
   
          $DATA/
          |–– food101/
          |   |–– food-101/
          |   |   |-- images/
          |   |   |-- meta/
          |   |   |-- ...
          |   |–– mask/
   
   - **StanfordCars**:
   
          $DATA/
          |–– stanford_cars/
          |   |–– stanford_cars/
          |   |   |-- cars_train/
          |   |   |-- cars_test/
          |   |   |-- ...
          |   |–– mask/

    - **OxfordPets**:
   
          $DATA/
          |–– oxford_pets/
          |   |–– oxford_pets/
          |   |   |-- annoations/
          |   |   |-- images/
          |   |   |-- ...
          |   |–– mask/
      
    - **Flowers102**:
   
          $DATA/
          |–– flowers-102/
          |   |–– oxford_flowers/
          |   |   |-- jpg/
          |   |   |-- cat_to_name.json
          |   |   |-- ...
          |   |–– mask/

    - **DTD**:
   
          $DATA/
          |–– dtd/
          |   |––dtd/
          |   |   |-- dtd/
          |   |   |   |-- images/
          |   |   |   |-- imdb/
          |   |   |   |-- ...
          |   |   |–– mask/

    - **EuroSAT**:
   
          $DATA/
          |–– eurosat/
          |   |–– eurosat/
          |   |   |-- 2750/
          |   |   |-- split_zhou_EuroSAT.json
          |   |–– mask/

    - **FGVC_Aircraft**:
   
          $DATA/
          |–– fgvc_aircraft/
          |   |–– fgvc-aircraft-2013b/
          |   |   |-- fgvc_aircraft
          |   |   |   |-- images/
          |   |   |   |-- families.txt
          |   |   |   |-- ...
          |   |   |–– mask/
          |   |   |–– ...

      
     - **SUN397**:
         ```
          $DATA/
          |–– sun-397/
          |   |–– sun397/
          |   |   |-- sun397/
          |   |   |   |-- a/
          |   |   |   |-- b/
          |   |   |   |-- ...
          |   |   |-- classnames.txt
          |   |   |-- split_zhou_SUN397.json
          |   |   |-- ...
          |   |–– mask/
        ```

      - **UCF101**:

        ```
          $DATA/
          |–– ucf101/
          |   |–– ucf101/
          |   |   |-- UCF-101-midframes/
          |   |   |-- ucfTrainTestlist/
          |   |   |-- split_zhou_UCF101.json
          |   |–– mask/
        ```
      
