TestBunche for Image Digit recognition
=======================================

Use the images in the folder 'img_digit_initial'to generate digit images with a fixed resolution and a set of parametres(Noise,Blur,rotation,....) that we need to specify in 'Param_file.txt' ,the generated images are put in a corresponding subfolder in the folder 'Test', next they are run in our chosen prediction script automatically (testbunche_original.py/testbunche_class.py/testbunche_conv.py/testbunche_final.py) to finally generates statistiques in the file 'Statistique_general.txt'.

Folders
========

##img_digit_initial

Contains the initial digit images ,to wich we are gonna apply different effects , and downsampling 

##Test

Contains the generated images after applying parametres and downsampling

scripts
=======

##BancTest.py

The main script that generates images and run them through the prediction script

##boocle.py

used to run 'BancTest.py' for a number of iterations that we spicify(in the code of the script)

##Param_file.txt

Contain the parametres that are used to generate images , and wich script is used for prediction

##Statistique_general.txt

Contain the generated statistiques , and reverse statistique


