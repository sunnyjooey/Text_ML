### NLP-based machine learning project classifying Twitter data

#### Things to install and download
sudo apt-get install software-properties-common
sudo apt-add-repository universe
sudo apt-get update
sudo apt install python3-pip
sudo apt-get install python3-tk
sudo apt-get install openjdk-8-jdk

pip3 install -r requirements.txt

#### In python
import info_extract #in data_preprocessing folder
info_extract.setupStanfordNLP()
import nltk
nltk.download('wordnet')
