# SIADS-capstone-building-damage-detection-TEAM-GOONERS-
We plan to assess levels/severity of damage for the damaged buildings but with MVP solution we at minimum plan to develop a model that can distinguish between damaged and non damaged buildings. 

proposal and questions we hope to answer or explore? 
Using pre and post disaster satellite imagery of disaster stricken areas, we will assess building damage to aid first responders in selecting/prioritizing relief work in the affected areas. Post disaster relief work in 24-48 hrs if carried out in most affected areas can improve chances of rescuing survivors and minimize human casualties. We hope our classification model can identify most damaged structures in the affected area so that it can be used by the disaster management team to direct resources for relief work.

What ethical challenges or concerns do we expect to encounter in this project? If there are potential concerns, how do we plan to mitigate them?
Our goal is identify damaged structures in the area due natural disaster and aide first responders to identify most impacted areas. Naturally, due to building density of cities, the model might have a bias in identifying mostly the city areas of the region and due to that rural areas (low density) might get neglected by the first responders. We will try to convey our model's prediction bias to the decision makers so that they can make necessary interventions in selecting first response target areas.

Data source: https://xview2.org/ 
From xView2 website first we needed to create an account and download the respective .tar files 
  Challenge training set (~7.8 GB)
  SHA1: b37a4ef4ee9c909e2b19d046e49d42ee3965714b
  Challenge test set (~2.6 GB)
  SHA1: 86ed3dba2f8d16ceceb75d451005054fefa9616f
  Challenge holdout set (~2.6 GB)
  SHA1: fe7f162f0895bfaff134cab3abc23872f38d17da
Data split was already provided between Train/Test/Hold (80/10/10)

Unzip above files to have the folder structure given below. 
Extract all folders in sample data folder. 

Update the config.yaml file in the configs folder with updated value for the new folder path.  

For ex. change "sampledata: ${hydra:runtime.cwd}/Sample data/" to ""sampledata: ${hydra:runtime.cwd}/Sample data/test" if you wish to run the inference.py on the test dataset.

Folder structure is as below:
- Train
     -- Images
     -- Labels
     -- Masks
- Test
     -- Images
     -- Labels
     -- Masks
- Hold
     -- Images
     -- Labels
     -- Masks

![image](https://user-images.githubusercontent.com/55030743/230085270-adcd5ec8-6c1e-4fa3-888f-a11fe453490e.png)

###How to run the Streamlit App:

Clone the github repository: 
$ git clone

Install dependencies: 
$ pip install -r requirements.txt

Run the streamlit app:
$ python -m streamlit run streamlit_app.py

Upload Pre and Post Disaster images:

Click “Assess Building Damage”

The app will present you with uploaded Pre disaster and Post disaster images along with the predicted damage prediction mask. 

