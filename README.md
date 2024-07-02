# StructureMetrics
Structure metrics are used to tell the authenticity of predicted structures. 
![](https://cdn.jsdelivr.net/gh/HuangJiaLian/DataBase0@master/uPic/2024-07-02-14-04-Intro.png)
We believe that the metrics like RDF and ADF can be used in both simulations and experiments if we agree on density functional theory. Therefore, distribution comes from the predicted structure of ML model by inputing real AFM images can be compared to the target distribution that originated from DFT calculations.

When comparing the a distribution Q to target distribution P, the KL divergence Div(P, Q) of these two distributions can be used. The KL divergence of  distribution P and Q, it measures how much information is lost when Q is used to approximate P.
![](https://cdn.jsdelivr.net/gh/HuangJiaLian/DataBase0@master/uPic/2024-07-02-15-18-HowToCompare.png)

Here the divergences can reveal the model performances on different input image set. 

- Div(P, Q\_1): Original Model performance on PPAFM images 
- Div(P, Q\_2): Original Model performance on Fake AFM images 
- Div(P, Q\_3): Original Model performance on experimental AFM images 
- Div(P, Q\_a): New Model performance on PPAFM images 
- Div(P, Q\_b): New Model performance on Fake AFM images
- Div(P, Q\_c): New Model performance on experimental AFM images 
