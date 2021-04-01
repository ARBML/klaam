# klaam
Arabic speech recognition and classification

 <p align="center"> 
 <img src = "https://raw.githubusercontent.com/ARBML/klaam/main/klaam_logo.PNG" width = "200px"/>
 </p>
 
 ## Installation 
 ```
 pip install klaam
 ```
 
 ## Usage 
 
 ```python
 from klaam import ClassificationModel
 model = ClassificationModel()
 model.classify('file.wav')
 
 from klaam import RecognitionModel
 model = RecognitionModel()
 model.transcribe('file.wav')
 ```
