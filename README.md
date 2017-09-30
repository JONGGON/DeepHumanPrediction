<center><image src="https://github.com/JONGGON/DeepHumanPrediction/blob/master/DeepHumanPrediction/HumanMotion_.png" width=1000 height=500></image></center>

>## ***Introduction*** 
*   
   `A place to 'post' the progress of my master's thesis.`

>## ***Progress(Related studies necessary for master 's thesis.)***

* ### **Where can I get the data?**

    * <https://accad.osu.edu/research/mocap/mocap_data.htm>
    * <http://mocapdata.com/>

* ### [**what is BVH file?**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/reference/BVH)

    <left><image src="https://github.com/JONGGON/DeepHumanPrediction/blob/master/DeepHumanPrediction/bvh.jpg" width=800 height=400></image></left>

* ### **Reading BVH file?**

    * [**Code**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/BVH_Reader)


    * [**PreProcessing the data Using Motionbuilder - Click on this Sentence and Learn**](https://www.youtube.com/watch?v=Apt-iN32cPo&list=PLtv0q3KQ5a9rKTl3v4qwmTY2VaXemwPu8)
        * `Since the skeleton information of 'ACCAD' and MOCAPDATA.COM data is different, each preprocessing is necessary.(ACCAD dataset -> Used in Progress , MOCAPDATA.COM dataset -> used in Master's Thesis)`

        * `Convert from C3D to BVH using MotionBuilder - It can be difficult because it uses a professional program called 'Motion Builder'.`
        
* ### **Writing BVH file?**

    * [**Code**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/BVH_Writer)

* ### [**Human Motion Prediction basic -Various attempts**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/DeepHumanPrediction) 
    * **Encoding + Decoding Structure**

        * [**Motion Prediction Simple - completed**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/DeepHumanPrediction/Motion_Prediction_Simple)

        * [**Motion Prediction encoding decoding - completed**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/DeepHumanPrediction/Motion_Prediction_encoding_decoding)

            * You have to learn one by one from easy data.(Please refer to the code.)
        * [**Motion Prediction encoding decoding using BidirectionalCell - completed**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/DeepHumanPrediction/Motion_Prediction_encoding_decoding_BidirectionalCell)

            * You have to learn one by one from easy data.(Please refer to the code.)

        * [**Motion Prediction encoding decoding : Joint angle to Cartesian coordinates using open source - Fixing**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/DeepHumanPrediction/Motion_Prediction_encoding_decoding_Joint_angle_to_Cartesian_coordinates)
            * You have to learn one by one from easy data.(Please refer to the code.)

        * [**Motion Prediction decoding encoding : using training_set and test_set - completed**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/DeepHumanPrediction/Motion_Prediction_encoding_decoding_training_set_and_test_set)
            * You have to learn one by one from easy data.(Please refer to the code.)   
        
        * [**Motion Prediction basic Seq2Seq sequencial learning version - completed**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/DeepHumanPrediction/Motion_Prediction_Seq2Seq_sequencialversion)
            * You have to learn one by one from easy data.(Please refer to the code.)    

        * [**Motion Prediction advanced Seq2Seq sequencial learning version - completed**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/DeepHumanPrediction/Motion_Prediction_advanced_Seq2Seq_sequencialversion)
            * You have to learn one by one from easy data.(Please refer to the code.) 

    * **Encoding + Decoding `Sequence to Sequence` Structure**
    
        * [**Motion Prediction advanced Seq2Seq batch learning version - completed**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/DeepHumanPrediction/Motion_Prediction_advanced_Seq2Seq_batchversion)

        * [**Motion Prediction Seq2Seq - Not Using Decoder Input - completed**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/DeepHumanPrediction/Motion_Prediction_Seq2Seq_No_Input_decoder)

>## ***Master's Thesis***

* ### [**Human motion for Master's Thesis**](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/Master_Thesis)

    * [***1-1. Motion Prediction(Generation) , Batch Learning , Input with noise , Tied weights , Training method = Test method***]()

    * [***1-2. Motion Prediction(Generation) , Batch Learning , Input with noise , Tied weights , Training method != Test method***](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/Master_Thesis/Motion_Generation)
        ```python
        '''Please read the notes below'''

        * about Network structure

            - If we share encoder and decoder weights, we can only use Residual Connection for 'One RNN Layer', 
            because 'ResidualCell' must have the same size of input and output.

            - The code for MultiLayer is not actually used, but is written for later use.

        * about Dataset
        
            - <Training data> : The training motion data (272) for 4 women, 68 motion data per person.

            - <Test data> : The Test motion data for 1 women, 68 motion data per person.
        ```
    * [**Neural Style for Motion Data : Data Augmentation - not yet**]()
    
        ```
        Let's change the structure more efficiently.
        ```
    * [***Human Motion Analysis consists of 3-Projects***](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/Master_Thesis/Human%20Motion%20Analysis)
        ```python
        - <Training data> : The training motion data (816) for 8 men and 4 women, 68 motion data per person.

        - <Test data> : The Test motion data (204) for 8 men and 4 women, 68 motion data per person.
        ```
        * [***1. Motion Classification***](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/Master_Thesis/Human%20Motion%20Analysis/Motion%20Classification)

        * [***2. Human Motion Pattern Classification***](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/Master_Thesis/Human%20Motion%20Analysis/Human%20Motion%20Pattern%20Classification) 


        * [***3-2. Motion Prediction(Generation) With more data***](https://github.com/JONGGON/DeepHumanPrediction/tree/master/DeepHumanPrediction/Code/Master_Thesis/Human%20Motion%20Analysis/Motion%20Generation)
            ```python
            Question? Can the network characterize motion data? Is it possible to generalize the Human motion?
            ```

>## ***Development environment***
* `window 10.1 64 bit` and `Ubuntu Linux 16.04.2 LTS` 
* `python verison : 3.6.1 , anaconda3 version : (4.4.0)` 
* `pycharm Community Edition 2017.2.2`

>## ***Dependencies*** 
* mxnet-0.11.1(`window`) , mxnet-0.11.1(`Linux`)
* tqdm -> (`progress`) , graphviz -> ( `Visualization` )