# Instruction

## Environment preparation
   
1. Install dependencies provided in `requirements.txt`
2. Set params in `params.py`
    
    You need to set params:
    ```
    main_dataset_directory // main directory with datasets
    main_results_directory // main directory where will be all results
    original_dataset_directory // path to original dataset
    augmentation_images_number // number of images that should be for single class after augmentation
    clusters // clusters number. Should be same as number of bricks classes
    ```
    depending on your preferences and data structure.
## Data preparation
    
1. Put all of your images in "original_dataset_directory". Folder structure should look like this:
 
     ```
    .../dataset
           /111
                -432432.jpg
                -R32ES32.jpg
                -321esa.jpg
                -wqe4324.jpg
                -erw5234.jpg
           /222
                -fdsr435.jpg
                -fdsfdst4.jpg
                -fdsfst543.jpg
                -fdsfs443.jpg
           /...
    ```
    where in a single directory like 111 are bricks from single class.

2. Start `augmentation_main.py` to augment dataset. Results will be in `images_main_directory` directory. Now in every directory should be at least `augmentation_images_number` images.
3. Now we need to rename all files. This step is required to next steps. Run `rename.py`.
4. Next step is to split dataset into train and test datasets. To do this we need to run `split_dataset.py`. By default data will be splitted in 80-20 proportions. If you want to change this proportions edit file `split_dataset.py`.

We have all data prepared. Now `images_main_directory` directory structure should look like this:

 ```
.../dataset_augmented
       /111
            /train
                -111_1.jpg
                -111_2.jpg
                -111_3.jpg
                ...
            /test
                -111_4.jpg
                -111_5.jpg
                ...
       /222
            /train
                -222_1.jpg
                -222_2.jpg
                -222_3.jpg
                ...
            /test
                -222_4.jpg
                -222_5.jpg
                ...
       /...
```

## First model training (step 0)

1. Create `models_directory` direcotry.
2. Run `features_extraction.py` to extract features from all images. This features will be selected as well. Result will be in `extracted_features_directory`.
3. Run `trainKMeans.py` to train K-Means model on extracted features in previous step.
4. Run `testKMeans.py` to test trained model. Now we will test on train dataset. We test on train dataset because in next steps we will need this prediction to train CNN model. To specify if we want to test on train or test dataset we can modify param `test_on_data`. 

Now we have finished testing step 0. Results should be visible in console and confusion matrix should be saved in `predicted_clusters_directory`. Also in `predicted_clusters_directory` directory should be subdirectories with created clusters. If we are not satisfied with the results we can train our CNN model. We will do this in step 1.

## CNN model training (step 1)

1. Comment `STEP 0` section and uncomment `STEP 1 ` section in `params.py`.
2. Run `clear_clusters.py` to simulate the removal of incorrectly assigned photos by the user. Attention! After this operation, changes will be made to the `previous_predicted_clusters_directory` directory. If you want to keep the results from the previous iteration, copy them to another directory. If algorithm have some problem there is a need of user cleaning directory.
3. Run `trainCNN.py` to train CNN model.
4. Now we can check how trained CNN deals with a given problem. To do this we need to run `testCNN.py`. Results will be in `predicted_classes_directory` direcotry. Additionally, we can check how our unsupervised model handled it, but in this manual we will base on the results of the supervised model in the next steps.

If we are still not satisfied with the results we can train our CNN model again. No we have larger dataset after step 1.

## CNN model training again (step 2)

1. Comment `STEP 1` section and uncomment `STEP 2 ` section in `params.py`.
2. Run `clear_clusters.py` to simulate the removal of incorrectly assigned photos by the user. Attention! After this operation, changes will be made to the `previous_predicted_clusters_directory` directory. If you want to keep the results from the previous iteration, copy them to another directory.  If algorithm have some problem there is a need of user cleaning directory.
3. Run `trainCNN.py` to train CNN model.
4. Now we can check how trained CNN deals with a given problem. To do this we need to run `testCNN.py`. This time parameter `test_on_data` is set to test. It means that we will test our solution on data that network didn't seen. Results will be in new `predicted_classes_directory` direcotry.

We can iteratively learn CNN model in next steps if we want.  
