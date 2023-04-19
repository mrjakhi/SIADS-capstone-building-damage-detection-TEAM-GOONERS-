# SIADS-capstone-building-damage-detection-TEAM-GOONERS-
We plan to assess levels/severity of damage for the damaged buildings but with MVP solution we at minimum plan to develop a model that can distinguish between damaged and non damaged buildings. 

- Natural disasters cause significant damage to our infrastructure.
- The ability to quickly assess the damage is critical for emergency responders
- Our goal is to apply ML to assess damage density in disaster-stricken areas using pre-disaster and post-disaster satellite imagery.


## Proposal and questions we hope to answer or explore? 
Using pre and post disaster satellite imagery of disaster stricken areas, we will assess building damage to aid first responders in selecting/prioritizing relief work in the affected areas. Post disaster relief work in 24-48 hrs if carried out in most affected areas can improve chances of rescuing survivors and minimize human casualties. We hope our classification model can identify most damaged structures in the affected area so that it can be used by the disaster management team to direct resources for relief work.

## What ethical challenges or concerns do we expect to encounter in this project? If there are potential concerns, how do we plan to mitigate them?
The model's purpose is to help with relief efforts. As they are closer to the scene, the relief activities are generally led by information gathered from local sources. As a result, their input should be prioritized over the model's forecast. 


We also want to emphasize that, in its current version, the model may not reliably anticipate minor and severe damage classifications, which the end user should be aware of. The model's performance will be influenced by cloud cover or picture quality, thus manual evaluation of the model's outputs should be performed before acting on the predictions. 


The model may mistakenly classify more buildings as damaged as a result of pixel leakage. This has the potential to impede rescue attempts in rural locations, which are far less crowded than metropolitan ones. We advise end customers to incorporate these factors into their decision-making process. 


## Data source: https://xview2.org/ 
From xView2 website first we needed to create an account and download the respective .tar files 

- Challenge training set (~7.8 GB)
  SHA1: b37a4ef4ee9c909e2b19d046e49d42ee3965714b
- Challenge test set (~2.6 GB)
  SHA1: 86ed3dba2f8d16ceceb75d451005054fefa9616f
- Challenge holdout set (~2.6 GB)
  SHA1: fe7f162f0895bfaff134cab3abc23872f38d17da
  
  
Data split was already provided between Train/Test/Hold (80/10/10)

Unzip above files to have the folder structure given below. 
Extract all folders in sample data folder. 

Update the config.yaml file in the configs folder with updated value for the new folder path.  

For eg. change "sampledata: ${hydra:runtime.cwd}/Sample data/" to ""sampledata: ${hydra:runtime.cwd}/Sample data/test" if you wish to run the inference.py on the test dataset.

Folder structure is as below:
- Train
     - Images
     - Labels
     - Masks
- Test
     - Images
     - Labels
     - Masks
- Hold
     - Images
     - Labels
     - Masks

![image](https://user-images.githubusercontent.com/55030743/233040342-c7934da5-01de-4a49-9eda-ab08167fd09f.png)

## Siamese-UNet architecture

![image](https://user-images.githubusercontent.com/55030743/233041865-187c7aac-cf24-4e88-8c09-a79ff52af617.png)


![image](https://user-images.githubusercontent.com/55030743/233042102-49487c42-661e-4cfd-ac07-3086392e85c1.png)

## How to run the Streamlit App:

Clone the github repository: 
$ git clone

Install dependencies: 
$ pip install -r requirements.txt

Run the streamlit app:
$ python -m streamlit run streamlit_app.py

Upload Pre and Post Disaster images:
![image](https://user-images.githubusercontent.com/55030743/233038963-4a62bc9f-0bff-41f4-a9ea-3642aeb077f6.png)


Click “Assess Building Damage”
![image](https://user-images.githubusercontent.com/55030743/233039062-3fb48b32-68a8-4946-8490-1fcc9cd8334b.png)


The app will present you with uploaded Pre disaster and Post disaster images along with the predicted damage prediction mask. 

