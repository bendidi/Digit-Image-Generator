#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division
import math,skimage,time,cv2,sys,os,configparser,ConfigParser
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy import misc
from skimage import *
from skimage import color
from pylab import *
import random
from ConfigParser import SafeConfigParser
from subprocess import check_output as qx




######################################fonction de generation de bruit #################################"
def noisy(image,var):
    noise = np.random.normal(0,var ** 0.5,image.shape)            
    noise = np.clip(noise, -1.0, 1.0)
    noise=img_as_ubyte(noise)
    out = cv2.add(image,noise)
    return out
######################### fonction pour changer la resolution de l'image en gardant le ratio ###############
def resize_contain(image,size):
    (h, l,nb) = image.shape
    img = np.zeros((size[1],size[0],3), np.uint8)
    img[:,:] = (255,255,255)
    a=int((l*h)/(size[0]*size[1]))
    rh=h-(int(h/size[1])*size[1])
    rl=l-(int(l/size[0])*size[0])
    t1,t2,testi,testj=0,0,0,0
    x1,x2=int(h/size[1]),int(l/size[0])
    for i in range(size[1]):
        for j in range(size[0]):
            px=[0,0,0]
            z1,z2,z3,z0=0,0,0,0
            if rh > 0:
                z1=1
            if t1==1 and testi != i:
                z0=1
                t1=0
            if rl > 0:
                z3=1
            if t2==1 and testj !=j:
                z2=1
                t2=0
            testi,testj=i,j
            if j==size[0]-1:
                testj=0
            for y in range((i*x1)+z0,((i+1)*x1)+z1):
                for x in range((j*x2)+z2,((j+1)*x2)+z3):
                    px=px+image[y,x]
            if z1==1:
                t1=1
            if z3==1:
                t2=1
            rl-=1
            if j==size[0]-1:
                rh-1
                rl=l-(int(l/size[0])*size[0])
            px=px/a
            img[i,j]=px
    return img

####################### fonction equalisation d'histogramme ####################
def GestionHist(image,bins,size):
    image = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
    (h,l) = image.shape
    pas_image=int(255/bins)
    pas_img=int((size[1]-size[0])/bins)
    lis=[]
    d={}
    for k in range(bins):
        lis.append(int(((size[0]+(k*pas_img))+(size[1]+((k+1)*pas_img)))/2))
        debut=k*pas_image
        fin=(k+1)*pas_image  
        d[debut,fin]=lis[k] 
    for i in range(h):
        for j in range(l):
            for (x,y) in d:
                if image[i,j] in range(x,y):
                    image[i,j]=d[x,y]
                    break
    image= cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
###################### fonction d'ajout de flou ##################
def Blur(img,size):
    output = cv2.blur(img, (size,size))### on filtre avec une matrice uniforme pour le blurring
    return output
####################### fonction d'ajout de flou optique  ################
def FlouOptique(image,sigma):
    img = ndimage.gaussian_filter(image, sigma)
    return img

############ fonction pour generer du flou cinetique #############
def MotionBlur(img,size,angle):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[:,int((size-1)/2)] = np.ones(size)
    kernel = kernel_motion_blur / size
    kernel = ndimage.rotate(kernel, angle)
    img = cv2.filter2D(img, -1, kernel) ## appliquer le le kernel comme filtre sur l'image ,ddepth=-1 pour avoir le meme depth que l'img src
    return img
#######################fonction rotation image ########################
def rotation(image,angle):
    image = ndimage.rotate(image,angle,reshape=False,mode="nearest")         
    return image
############ database d'image de digit initial ###################

os.chdir("img_digit_initial")
img=[]
for i in range(0,10):    
    img.append(cv2.imread('{}.jpg'.format(i)))
os.chdir(os.getcwd()+'/..')

####### les parametre acquis depuis le fichier Param_Ile.txt ###############

config=configparser.ConfigParser()
config.read("Param_File.txt")
path=str(config["parametre"]["nom de la version du script"])
b3=float(config["parametre"]["gaussian noise(variance dans [0.001,0.01])"])
a1=int(config["parametre"]["image rotaion(angle de rotation)"])
b2=int(config["parametre"]["blur(taille_du_filtre_en_pixel)"])
a3=int(config["parametre"]["flou optique(taille_filtre_en_pixel)"])
c3=config["parametre"]["motion_blur(taille_du_filtre_en_pixel,angle)"]
a6=config["parametre"]["Gestion d histogramme(bins,range[x,y])"]
a7=config["parametre"]["nombre d'image par resolution(7x11, 6x9, 5x8, 4x6, 3x4, 2x3)(dans l'ordre)"]
v1=config["customaize image resolution"]["largeur de l'image(en pixel)"]
v2=config["customaize image resolution"]["hauteur de l'image(en pixel)"]
v3=config["customaize image resolution"]["______nombre d'image________"]
a8=[]
a2=[]
x=[]
y=[]
n=[]
a66=[]
lenght=len(v1.split(","))
for i in range(0,lenght):    
    x.append(int(v1.split(",")[i]))
    y.append(int(v2.split(",")[i]))
    n.append(int(v3.split(",")[i]))
for i in range(0,3):
    a66.append(int(a6.split(",")[i]))        
for i in range(0,6):
    a8.append(int(a7.split(",")[i]))
a2.append(int(c3.split(",")[0])) 
a2.append(int(c3.split(",")[1]))        
###########fonction tup() pour calcul resolution #############

def tup(i):
    """docstring for tup"""
    return (i,int(i*8/5))
#########Creer le dossier de test  ##################
t=time.strftime("%d-%m-%Y--%H:%M:%S")
if os.path.isdir("Tests/"+path):
    os.mkdir("Tests/"+path+"/[{}]".format(t))
else:
    os.makedirs("Tests/"+path+"/[{}]".format(t))
os.chdir("Tests/"+path+"/[{}]".format(t))
########## creation du fichier parmetre.txt dans chaque dossier test ############    
config1 =configparser.ConfigParser()
config1["Parametre"]={"Nombre d'image par resolution(7x11, 6x9, 5x8, 4x6, 3x4, 2x3)(dans l'ordre)":a8, 
                      "image rotaion(angle de rotation)":a1,
                      "Gaussian Noise(Variance dans [0.001,0.01])":b3,
                      "blur(taille_du_filtre_en_pixel)":b2,
                      "flou optique(taille_filtre_en_pixel)":a3,                      
                      "motion_blur(taille_du_filtre_en_pixel,angle)":a2,
                      "Gestion d histogramme(bins,range[x,y])":a66,}
config1["customaize image resolution"]={"largeur de l'image(en pixel)":x,
                                        "hauteur de l'image(en pixel)":y,
                                        "______nombre d'image________":n,}
with open("ParamFile.txt","w") as configfile1:
    config1.write(configfile1)
################## boucle pour creer les dossier {0...9} dans chaque dossier test et generer les images selon les params #########
if __name__=='__main__':
    for k in range(0,10):
        if a2[0] > 1:
            img[k]=MotionBlur(img[k],a2[0],a2[1]) 
#            plt.imshow(img[k])
#            plt.show()
        if  b2 > 1:
            img[k]=Blur(img[k],b2)
        if  a3 > 1:
            img[k]=FlouOptique(img[k],a3)

        if  b3 != 0:
            img[k]=noisy(img[k],b3)
        if a66[0] != 0 and a66[2] != 0:
            img[k]=GestionHist(img[k],a66[0],(a66[1],a66[2]))
        if a1 != 0:
            img[k]=rotation(img[k],a1)          
        os.mkdir("{}".format(k))
        os.chdir("{}".format(k))
        img1=img[k]
        lx,ly,nx=img1.shape
        for d in range(0,a8[0]):
            x1=random.randint(0,3)
            x2=random.randint(0,3)
            x3=random.randint(0,3)
            x4=random.randint(0,3)            
            imgx= img1[x1:lx-x2, x3:ly-x4]            
            imgx=resize_contain(imgx,(7,11))
            cv2.imwrite("image_7x11_{}.jpg".format(d),imgx)
        for d in range(0,a8[1]):
            x1=random.randint(0,3)
            x2=random.randint(0,3)
            x3=random.randint(0,3)
            x4=random.randint(0,3)   
            imgx= img1[x1:lx-x2, x3:ly-x4]            
            imgx=resize_contain(imgx,(6,9))
            cv2.imwrite("image_6x9_{}.jpg".format(d),imgx)
        for d in range(0,a8[2]):
            x1=random.randint(0,3)
            x2=random.randint(0,3)
            x3=random.randint(0,3)
            x4=random.randint(0,3)   
            imgx= img1[x1:lx-x2, x3:ly-x4]            
            imgx=resize_contain(imgx,(5,8))
            cv2.imwrite("image_5x8_{}.jpg".format(d),imgx)
        for d in range(0,a8[3]):
            x1=random.randint(0,5)
            x2=random.randint(0,5)
            x3=random.randint(0,5)
            x4=random.randint(0,5) 
            imgx= img1[x1:lx-x2, x3:ly-x4]            
            imgx=resize_contain(imgx,(4,6))
            cv2.imwrite("image_4x6_{}.jpg".format(d),imgx)
        for d in range(0,a8[4]):
            x1=random.randint(0,3)
            x2=random.randint(0,3)
            x3=random.randint(0,3)
            x4=random.randint(0,3)  
            imgx= img1[x1:lx-x2, x3:ly-x4]            
            imgx=resize_contain(imgx,(3,4))
            cv2.imwrite("image_3x4_{}.jpg".format(d),imgx)
        for d in range(0,a8[5]):
            x1=random.randint(0,3)
            x2=random.randint(0,3)
            x3=random.randint(0,3)
            x4=random.randint(0,3)  
            imgx= img1[x1:lx-x2, x3:ly-x4]            
            imgx=resize_contain(imgx,(2,3))
            cv2.imwrite("image_2x3_{}.jpg".format(d),imgx)
        for i in range(0,lenght):
            if x[i] != 0 and y[i] != 0:
                for d in range(0,n[i]):
                    x1=random.randint(0,3)
                    x2=random.randint(0,3)
                    x3=random.randint(0,3)
                    x4=random.randint(0,3)  
                    lx,ly,nx=img1.shape
                    imgx= img1[x1:lx-x2, x3:ly-x4]            
                    imgx=resize_contain(imgx,(x[i],y[i]))
                    cv2.imwrite("image_{}x{}_{}.jpg".format(x[i],y[i],d),imgx)      
        os.chdir(os.getcwd()+'/..') ## return to [time] file
    os.chdir(os.getcwd()+'/..') ## return to version file 
    os.chdir(os.getcwd()+'/..') ## return to tests file
    os.chdir(os.getcwd()+'/..') ## return to TestBunche
    os.chdir(os.getcwd()+'/..') ## return to verfin
    if int(path)!=0:
        output = qx(['python', path,'-i','/home/ouail/ver_fin/TestBunche/Tests/'+path+'/[{}]/'.format(t)])
        a=[]
        print(output)
        b=output.split(",")
        for i in range(0,10):
            a.append(int(b[i][1]))
        taux=0
        for i in range(0,10):
            if a[i]==i:
                taux=taux+1
        os.chdir("TestBunche")
        configset = ConfigParser.RawConfigParser()
        configset.add_section('Resultat du test du {}'.format(t))
        configset.set('Resultat du test du {}'.format(t),'nombre total d image pour chaque digit', sum(n)+sum(a8))
        if a1 != 0:
            configset.set('Resultat du test du {}'.format(t),'image rotaion(angle de rotation)', a1)
        if  b3 != 0:
            configset.set('Resultat du test du {}'.format(t),'Gaussian Noise(Variance dans [0.001,0.01])', b3)
        if  b2 > 1:
            configset.set('Resultat du test du {}'.format(t),'blur(taille_du_filtre_en_pixel)', b2)
        if  a3 > 1:
            configset.set('Resultat du test du {}'.format(t),'flou optique(taille_filtre_en_pixel)', a3)
        if a2[0] > 1:
            configset.set('Resultat du test du {}'.format(t),'motion_blur(taille_du_filtre_en_pixel,angle)', a2)
        if a66[0] != 0:
            configset.set('Resultat du test du {}'.format(t),'Gestion d histogramme(bins,range[x,y])', a66)
        for i in range(10):
            configset.set('Resultat du test du {}'.format(t),'Prediction des images du digit {}'.format(i), a[i])
        configset.set('Resultat du test du {}'.format(t),'nombre de digit correctement predit', taux)
        with open('Resultat_par_test.txt', 'a') as configfile:
            configset.write(configfile)

        parser = SafeConfigParser()
        parser.read('Statistique_General.txt')
        Nx= int(parser.get("Statistique general des tests", "Nombre des tests effectués jusqu'à present"))
        l=[]
        for i in range(10):
            l.append(parser.get('Statistique general des tests', 'pourcentage of prediction of {} as(0,9)'.format(i)))
        m=[]
        for k in range(0,10):
            zut=[]
            l[k]=l[k].split(",")
            zut.append(float(l[k][0].split("[")[1]))

            for i in range(1,9):
                zut.append(float(l[k][i]))
            zut.append(float(l[k][9].split("]")[0]))
            for i in range(0,10):
                zut[i]=float((zut[i]*(Nx-1)/100))
            m.append(zut)
        true=[0,1,2,3,4,5,6,7,8,9]
        if Nx != 0:
            for k in range(0,10):
                for z in range(0,10):
                    if a[k]==true[z]:
                        m[k][z]=m[k][z]+1
            for k in range(0,10):
                for i in range(0,10):
                    m[k][i]=float((m[k][i]*100)/Nx)
                    m[k][i]=round(m[k][i],2)                
        else:
            for k in range(0,10):
                for i in range(0,10):
                    m[k][i]=0
            for k in range(0,10):
                for z in range(0,10):
                    if a[k]==true[z]:
                        m[k][z]=(m[k][z]+1)*100        
        Nx=Nx+1
        pred=[]  
        rev=[]      
        for i in range(0,10):
            pred.append(m[i][i])
        def sommecol(i):
            res=0.01
            for k in range(0,10):
                res=res+m[k][i]  
            return res
        for k in range(0,10):
            rev1=[]
            for i in range(0,10):
                rev1.append(round((m[k][i]*100)/sommecol(i),2))
            rev.append(rev1)
        dl=[]        
        for k in range(0,10):
            dl1=[]
            for i in range(0,10):
                dl1.append(rev[i][k])
            dl.append(dl1)
                    
        configg =ConfigParser.RawConfigParser()
        configg.add_section('Statistique general des tests')
        configg.set("Statistique general des tests", "Nombre des tests effectués jusqu'à present", Nx)
        configg.set("Statistique general des tests", "pourcentage total prediction", pred)
        configg.set('Statistique general des tests', '--------------------------------------', '   0|    1|   2|   3|   4|   5|   6|   7|   8|   9|')
        for i in range(10):
            configg.set('Statistique general des tests', 'pourcentage of prediction of {i} as(0,9)'.format(i), m[i])
        configg.add_section('reverse stats')
        for i in range(10):
            configg.set('reverse stats', 'if {} is predicted then original is:'.format(i), dl[i])
        configg.add_section('Comment lire le tableau')
        configg.set('Comment lire le tableau', '1', 'Ce tableau est lis de la maniere suivante le pourcentage de prediction de 0 est de X en tant que 0 , X en tant que 1 , X en tant que 2 .....')
        configg.set('Comment lire le tableau', '2', 'Le tableau est lis ligne par ligne ')
        configg.set('Comment lire le tableau', '3', 'La diagonal represente le pourcentage de la prediction correct de chaque digit')
        configg.set("Comment lire le tableau", "4", "pour reset les statistique veillez mettre le nombre de test effectué jusqu'a present a 0" )
        configg.add_section('Parametre du dernier test du {}'.format(t))
        configg.set('Parametre du dernier test du {}'.format(t),'nombre total d image pour chaque digit', sum(n)+sum(a8))
        if a1 != 0:
            configg.set('Parametre du dernier test du {}'.format(t),'image rotaion(angle de rotation)', a1)
        if  b3 != 0:
            configg.set('Parametre du dernier test du {}'.format(t),'Gaussian Noise(Variance dans [0.001,0.01])', b3)
        if  b2 > 1:
            configg.set('Parametre du dernier test du {}'.format(t),'blur(taille_du_filtre_en_pixel)', b2)
        if  a3 > 1:
            configg.set('Parametre du dernier test du {}'.format(t),'flou optique(taille_filtre_en_pixel)', a3)
        if a2[0] > 1:
            configg.set('Parametre du dernier test du {}'.format(t),'motion_blur(taille_du_filtre_en_pixel,angle)', a2)
        if a66[0] != 0 and a66[2] != 0:
            configg.set('Parametre du dernier test du {}'.format(t),'Gestion d histogramme(bins,range[x,y])', a66)
        configg.add_section('Resultat du dernier test du {}'.format(t))
        for i in range(10):
            configg.set('Resultat du dernier test du {}'.format(t),'Prediction des images du digit {}'.format(i), a[i])
        configg.set('Resultat du dernier test du {}'.format(t),'nombre de digit correctement predit', taux)
        with open('Statistique_General.txt', 'w') as configfile:
            configg.write(configfile)
        os.chdir("Tests/"+path+"/[{}]".format(t))
        configsett = ConfigParser.RawConfigParser()
        configsett.add_section('Resultat du test du {}'.format(t))
        for i in range(10):
            configsett.set('Resultat du test du {}'.format(t),'Prediction des images du digit {}'.format(i), a[i])
        with open('Result_File.txt', 'w') as configfile:
            configsett.write(configfile)
        os.chdir(os.getcwd()+'/..')
        os.chdir(os.getcwd()+'/..')
        os.chdir(os.getcwd()+'/..')









